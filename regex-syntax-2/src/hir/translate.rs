use std::cell::{Cell, RefCell};
use std::result;

use ast::{self, Ast, Span, Visitor};
use either::Either;
use hir::{self, Error, ErrorKind, Hir};
use unicode::{self, ClassQuery};

type Result<T> = result::Result<T, Error>;

/// A builder for constructing an AST->HIR translator.
#[derive(Clone, Debug)]
pub struct TranslatorBuilder {
    allow_invalid_utf8: bool,
}

impl Default for TranslatorBuilder {
    fn default() -> TranslatorBuilder {
        TranslatorBuilder::new()
    }
}

impl TranslatorBuilder {
    /// Create a new translator builder with a default c onfiguration.
    pub fn new() -> TranslatorBuilder {
        TranslatorBuilder {
            allow_invalid_utf8: false,
        }
    }

    /// Build a translator using the current configuration.
    pub fn build(&self) -> Translator {
        Translator {
            stack: RefCell::new(vec![]),
            flags: Cell::new(Flags::default()),
            allow_invalid_utf8: self.allow_invalid_utf8,
        }
    }

    /// When enabled, translation will permit the construction of a regular
    /// expression that may match invalid UTF-8.
    ///
    /// When disabled (the default), the translator is guaranteed to produce
    /// an expression that will only ever match valid UTF-8 (otherwise, the
    /// translator will return an error).
    pub fn allow_invalid_utf8(
        &mut self,
        yes: bool,
    ) -> &mut TranslatorBuilder {
        self.allow_invalid_utf8 = yes;
        self
    }
}

/// A translator maps abstract syntax to a high level intermediate
/// representation.
///
/// A translator may be benefit from reuse. That is, a translator can translate
/// many abstract syntax trees.
///
/// The lifetime `'a` refers to the lifetime of the abstract syntax tree that
/// is being translated.
#[derive(Clone, Debug)]
pub struct Translator {
    /// Our call stack, but on the heap.
    stack: RefCell<Vec<HirFrame>>,
    /// The current flag settings.
    flags: Cell<Flags>,
    /// Whether we're allowed to produce HIR that can match arbitrary bytes.
    allow_invalid_utf8: bool,
}

impl Translator {
    /// Create a new translator using the default configuration.
    pub fn new() -> Translator {
        TranslatorBuilder::new().build()
    }

    /// Translate the given abstract syntax tree (AST) into a high level
    /// intermediate representation (HIR).
    ///
    /// If there was a problem doing the translation, then an HIR-specific
    /// error is returned.
    ///
    /// The original pattern string used to produce the Ast should also be
    /// provided. While the translator does not use during any correct
    /// translation, it is used for error reporting.
    pub fn translate(&mut self, pattern: &str, ast: &Ast) -> Result<Hir> {
        ast::visit(ast, TranslatorI::new(self, pattern))
    }
}

/// An HirFrame is a single stack frame, represented explicitly, which is
/// created for each item in the Ast that we traverse.
///
/// Note that technically, this type doesn't represent our entire stack
/// frame. In particular, the Ast visitor represents any state associated with
/// traversing the Ast itself.
#[derive(Clone, Debug)]
enum HirFrame {
    /// An arbitrary HIR expression. These get pushed whenever we hit a base
    /// case in the Ast. They get popped after an inductive (i.e., recursive)
    /// step is complete.
    Expr(Hir),
    /// A Unicode character class. This frame is mutated as we descend into
    /// the Ast of a character class (which is itself its own mini recursive
    /// structure).
    ClassUnicode(hir::ClassUnicode),
    /// A byte-oriented character class. This frame is mutated as we descend
    /// into the Ast of a character class (which is itself its own mini
    /// recursive structure).
    ///
    /// Byte character classes are created when Unicode mode (`u`) is disabled.
    /// If `allow_invalid_utf8` is disabled (the default), then a byte
    /// character is only permitted to match ASCII text.
    ClassBytes(hir::ClassBytes),
    /// This is pushed on to the stack upon first seeing any kind of group,
    /// indicated by parentheses (including non-capturing groups). It is popped
    /// upon leaving a group.
    Group {
        /// The old active flags, if any, when this group was opened.
        ///
        /// If this group sets flags, then the new active flags are set to the
        /// result of merging the old flags with the flags introduced by this
        /// group.
        ///
        /// When this group is popped, the active flags should be restored to
        /// the flags set here.
        ///
        /// The "active" flags correspond to whatever flags are set in the
        /// Translator.
        old_flags: Option<Flags>,
    },
    /// This is pushed whenever a concatenation is observed. After visiting
    /// every sub-expression in the concatenation, the translator's stack is
    /// popped until it sees a Concat frame.
    Concat,
    /// This is pushed whenever an alternation is observed. After visiting
    /// every sub-expression in the alternation, the translator's stack is
    /// popped until it sees an Alternation frame.
    Alternation,
}

impl HirFrame {
    /// Assert that the current stack frame is an Hir expression and return it.
    fn unwrap_expr(self) -> Hir {
        match self {
            HirFrame::Expr(expr) => expr,
            _ => panic!("tried to unwrap expr from HirFrame, got: {:?}", self)
        }
    }

    /// Assert that the current stack frame is a Unicode class expression and
    /// return it.
    fn unwrap_class_unicode(self) -> hir::ClassUnicode {
        match self {
            HirFrame::ClassUnicode(cls) => cls,
            _ => panic!("tried to unwrap Unicode class \
                         from HirFrame, got: {:?}", self)
        }
    }

    /// Assert that the current stack frame is a byte class expression and
    /// return it.
    fn unwrap_class_bytes(self) -> hir::ClassBytes {
        match self {
            HirFrame::ClassBytes(cls) => cls,
            _ => panic!("tried to unwrap byte class \
                         from HirFrame, got: {:?}", self)
        }
    }

    /// Assert that the current stack frame is a group indicator and return
    /// its corresponding flags (the flags that were active at the time the
    /// group was entered) if they exist.
    fn unwrap_group(self) -> Option<Flags> {
        match self {
            HirFrame::Group { old_flags } => old_flags,
            _ => panic!("tried to unwrap group from HirFrame, got: {:?}", self)
        }
    }
}

impl<'t, 'p> Visitor for TranslatorI<'t, 'p> {
    type Output = Hir;
    type Err = Error;

    fn finish(self) -> Result<Hir> {
        Ok(self.pop().unwrap().unwrap_expr())
    }

    fn visit_pre(&mut self, ast: &Ast) -> Result<()> {
        match *ast {
            Ast::Class(ast::Class::Bracketed(_)) => {
                if self.flags().unicode() {
                    let cls = hir::ClassUnicode::empty();
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let cls = hir::ClassBytes::empty();
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            Ast::Group(ref x) => {
                let old_flags = x.flags().map(|ast| self.set_flags(ast));
                self.push(HirFrame::Group {
                    old_flags: old_flags,
                });
            }
            Ast::Concat(ref x) if x.asts.is_empty() => {}
            Ast::Concat(ref x) => {
                self.push(HirFrame::Concat);
            }
            Ast::Alternation(ref x) if x.asts.is_empty() => {}
            Ast::Alternation(ref x) => {
                self.push(HirFrame::Alternation);
            }
            _ => {}
        }
        Ok(())
    }

    fn visit_post(&mut self, ast: &Ast) -> Result<()> {
        match *ast {
            Ast::Empty(_) => {
                self.push(HirFrame::Expr(Hir::Empty));
            }
            Ast::Flags(ref x) => {
                self.set_flags(&x.flags);
            }
            Ast::Literal(ref x) => {
                self.push(HirFrame::Expr(try!(self.hir_literal(x))));
            }
            Ast::Dot(span) => {
                self.push(HirFrame::Expr(try!(self.hir_dot(span))));
            }
            Ast::Assertion(ref x) => {
                self.push(HirFrame::Expr(self.hir_assertion(x)));
            }
            Ast::Class(ast::Class::Perl(ref x)) => {
                if self.flags().unicode() {
                    let cls = self.hir_perl_unicode_class(x);
                    let hcls = hir::Class::Unicode(cls);
                    self.push(HirFrame::Expr(Hir::Class(hcls)));
                } else {
                    let cls = self.hir_perl_byte_class(x);
                    let hcls = hir::Class::Bytes(cls);
                    self.push(HirFrame::Expr(Hir::Class(hcls)));
                }
            }
            Ast::Class(ast::Class::Unicode(ref x)) => {
                let cls = hir::Class::Unicode(try!(self.hir_unicode_class(x)));
                self.push(HirFrame::Expr(Hir::Class(cls)));
            }
            Ast::Class(ast::Class::Bracketed(ref ast_cls)) => {
                if self.flags().unicode() {
                    let mut cls = self.pop().unwrap().unwrap_class_unicode();
                    self.unicode_fold_and_negate(ast_cls.negated, &mut cls);
                    let expr = Hir::Class(hir::Class::Unicode(cls));
                    self.push(HirFrame::Expr(expr));
                } else {
                    let mut cls = self.pop().unwrap().unwrap_class_bytes();
                    try!(self.bytes_fold_and_negate(ast_cls, &mut cls));

                    let expr = Hir::Class(hir::Class::Bytes(cls));
                    self.push(HirFrame::Expr(expr));
                }
            }
            Ast::Repetition(ref x) => {
                let expr = self.pop().unwrap().unwrap_expr();
                self.push(HirFrame::Expr(self.hir_repetition(x, expr)));
            }
            Ast::Group(ref x) => {
                let expr = self.pop().unwrap().unwrap_expr();
                if let Some(flags) = self.pop().unwrap().unwrap_group() {
                    self.trans().flags.set(flags);
                }
                self.push(HirFrame::Expr(self.hir_group(x, expr)));
            }
            Ast::Concat(_) => {
                let mut exprs = vec![];
                while let Some(HirFrame::Expr(expr)) = self.pop() {
                    exprs.push(expr);
                }
                exprs.reverse();
                self.push(HirFrame::Expr(Hir::concat(exprs)));
            }
            Ast::Alternation(_) => {
                let mut exprs = vec![];
                while let Some(HirFrame::Expr(expr)) = self.pop() {
                    exprs.push(expr);
                }
                exprs.reverse();
                self.push(HirFrame::Expr(Hir::alternation(exprs)));
            }
        }
        Ok(())
    }

    fn visit_class_set_item_pre(
        &mut self,
        ast: &ast::ClassSetItem,
    ) -> Result<()> {
        match *ast {
            ast::ClassSetItem::Bracketed(_) => {
                if self.flags().unicode() {
                    let cls = hir::ClassUnicode::empty();
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let cls = hir::ClassBytes::empty();
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            // We needn't handle the Union case here since the visitor will
            // do it for us.
            _ => {}
        }
        Ok(())
    }

    fn visit_class_set_item_post(
        &mut self,
        ast: &ast::ClassSetItem,
    ) -> Result<()> {
        match *ast {
            ast::ClassSetItem::Empty(_) => {}
            ast::ClassSetItem::Literal(ref x) => {
                if self.flags().unicode() {
                    let mut cls = self.pop().unwrap().unwrap_class_unicode();
                    cls.push(hir::ClassRangeUnicode::new(x.c, x.c));
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let mut cls = self.pop().unwrap().unwrap_class_bytes();
                    let byte = try!(self.class_literal_byte(x));
                    cls.push(hir::ClassRangeBytes::new(byte, byte));
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            ast::ClassSetItem::Range(ref x) => {
                if self.flags().unicode() {
                    let mut cls = self.pop().unwrap().unwrap_class_unicode();
                    cls.push(hir::ClassRangeUnicode::new(x.start.c, x.end.c));
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let mut cls = self.pop().unwrap().unwrap_class_bytes();
                    let start = try!(self.class_literal_byte(&x.start));
                    let end = try!(self.class_literal_byte(&x.end));
                    cls.push(hir::ClassRangeBytes::new(start, end));
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            ast::ClassSetItem::Ascii(ref x) => {
                if self.flags().unicode() {
                    let mut cls = self.pop().unwrap().unwrap_class_unicode();
                    for &(s, e) in ascii_class(&x.kind) {
                        cls.push(hir::ClassRangeUnicode::new(s, e));
                    }
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let mut cls = self.pop().unwrap().unwrap_class_bytes();
                    for &(s, e) in ascii_class(&x.kind) {
                        cls.push(hir::ClassRangeBytes::new(s as u8, e as u8));
                    }
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            ast::ClassSetItem::Unicode(ref x) => {
                let xcls = try!(self.hir_unicode_class(x));
                let mut cls = self.pop().unwrap().unwrap_class_unicode();
                cls.union(&xcls);
                self.push(HirFrame::ClassUnicode(cls));
            }
            ast::ClassSetItem::Perl(ref x) => {
                if self.flags().unicode() {
                    let xcls = self.hir_perl_unicode_class(x);
                    let mut cls = self.pop().unwrap().unwrap_class_unicode();
                    cls.union(&xcls);
                    self.push(HirFrame::ClassUnicode(cls));
                } else {
                    let xcls = self.hir_perl_byte_class(x);
                    let mut cls = self.pop().unwrap().unwrap_class_bytes();
                    cls.union(&xcls);
                    self.push(HirFrame::ClassBytes(cls));
                }
            }
            ast::ClassSetItem::Bracketed(ref ast_cls) => {
                if self.flags().unicode() {
                    let mut cls1 = self.pop().unwrap().unwrap_class_unicode();
                    self.unicode_fold_and_negate(ast_cls.negated, &mut cls1);

                    let mut cls2 = self.pop().unwrap().unwrap_class_unicode();
                    cls2.union(&cls1);
                    self.push(HirFrame::ClassUnicode(cls2));
                } else {
                    let mut cls1 = self.pop().unwrap().unwrap_class_bytes();
                    try!(self.bytes_fold_and_negate(ast_cls, &mut cls1));

                    let mut cls2 = self.pop().unwrap().unwrap_class_bytes();
                    cls2.union(&cls1);
                    self.push(HirFrame::ClassBytes(cls2));
                }
            }
            // This is handled automatically by the visitor.
            ast::ClassSetItem::Union(ref union) => {}
        }
        Ok(())
    }

    fn visit_class_set_binary_op_pre(
        &mut self,
        op: &ast::ClassSetBinaryOp,
    ) -> Result<()> {
        if self.flags().unicode() {
            let cls = hir::ClassUnicode::empty();
            self.push(HirFrame::ClassUnicode(cls));
        } else {
            let cls = hir::ClassBytes::empty();
            self.push(HirFrame::ClassBytes(cls));
        }
        Ok(())
    }

    fn visit_class_set_binary_op_in(
        &mut self,
        _op: &ast::ClassSetBinaryOp,
    ) -> Result<()> {
        if self.flags().unicode() {
            let cls = hir::ClassUnicode::empty();
            self.push(HirFrame::ClassUnicode(cls));
        } else {
            let cls = hir::ClassBytes::empty();
            self.push(HirFrame::ClassBytes(cls));
        }
        Ok(())
    }

    fn visit_class_set_binary_op_post(
        &mut self,
        op: &ast::ClassSetBinaryOp,
    ) -> Result<()> {
        use ast::ClassSetBinaryOpKind::*;

        if self.flags().unicode() {
            let mut rhs = self.pop().unwrap().unwrap_class_unicode();
            let mut lhs = self.pop().unwrap().unwrap_class_unicode();
            if self.flags().case_insensitive() {
                rhs.case_fold_simple();
                lhs.case_fold_simple();
            }
            match op.kind {
                Intersection => lhs.intersect(&rhs),
                Difference => lhs.difference(&rhs),
                SymmetricDifference => lhs.symmetric_difference(&rhs),
            }
            self.push(HirFrame::ClassUnicode(lhs));
        } else {
            let mut rhs = self.pop().unwrap().unwrap_class_bytes();
            let mut lhs = self.pop().unwrap().unwrap_class_bytes();
            if self.flags().case_insensitive() {
                rhs.case_fold_simple();
                lhs.case_fold_simple();
            }
            match op.kind {
                Intersection => lhs.intersect(&rhs),
                Difference => lhs.difference(&rhs),
                SymmetricDifference => lhs.symmetric_difference(&rhs),
            }
            self.push(HirFrame::ClassBytes(lhs));
        }
        Ok(())
    }
}

/// The internal implementation of a translator.
///
/// This type is responsible for carrying around the original pattern string,
/// which is not tied to the internal state of a translator.
///
/// A TranslatorI exists for the time it takes to translate a single Ast.
#[derive(Clone, Debug)]
struct TranslatorI<'t, 'p> {
    trans: &'t Translator,
    pattern: &'p str,
}

impl<'t, 'p> TranslatorI<'t, 'p> {
    /// Build a new internal translator.
    fn new(trans: &'t Translator, pattern: &'p str) -> TranslatorI<'t, 'p> {
        TranslatorI { trans: trans, pattern: pattern }
    }

    /// Return a reference to the underlying translator.
    fn trans(&self) -> &Translator {
        &self.trans
    }

    /// Push the given frame on to the call stack.
    fn push(&self, frame: HirFrame) {
        self.trans().stack.borrow_mut().push(frame);
    }

    /// Pop the top of the call stack. If the call stack is empty, return None.
    fn pop(&self) -> Option<HirFrame> {
        self.trans().stack.borrow_mut().pop()
    }

    /// Create a new error with the given span and error type.
    fn error(&self, span: Span, kind: ErrorKind) -> Error {
        Error { kind: kind, pattern: self.pattern.to_string(), span: span }
    }

    /// Return a copy of the active flags.
    fn flags(&self) -> Flags {
        self.trans().flags.get()
    }

    /// Set the flags of this translator from the flags set in the given AST.
    /// Then, return the old flags.
    fn set_flags(&self, ast_flags: &ast::Flags) -> Flags {
        let old_flags = self.flags();
        let mut new_flags = Flags::from_ast(ast_flags);
        new_flags.merge(&old_flags);
        self.trans().flags.set(new_flags);
        old_flags
    }

    fn hir_literal(&self, lit: &ast::Literal) -> Result<Hir> {
        let ch = match try!(self.literal_to_char(lit)) {
            Either::Right(byte) => return Ok(Hir::Literal(byte)),
            Either::Left(ch) => ch,
        };
        if self.flags().case_insensitive() {
            self.hir_from_char_case_insensitive(lit.span, ch)
        } else {
            self.hir_from_char(lit.span, ch)
        }
    }

    /// Convert an Ast literal to its scalar representation.
    ///
    /// When Unicode mode is enabled, then this always succeeds and returns a
    /// `char` (Unicode scalar value).
    ///
    /// When Unicode mode is disabled, then a raw byte is returned. If that
    /// byte is not ASCII and invalid UTF-8 is not allowed, then this returns
    /// an error.
    fn literal_to_char(&self, lit: &ast::Literal) -> Result<Either<char, u8>> {
        if self.flags().unicode() {
            return Ok(Either::Left(lit.c));
        }
        let byte = match lit.byte() {
            None => return Ok(Either::Left(lit.c)),
            Some(byte) => byte,
        };
        if !self.trans().allow_invalid_utf8 && byte > 0x7F {
            return Err(self.error(lit.span, ErrorKind::InvalidUtf8));
        }
        Ok(Either::Right(byte))
    }

    fn hir_from_char(&self, span: Span, c: char) -> Result<Hir> {
        if !self.flags().unicode() && c.len_utf8() > 1 {
            return Err(self.error(span, ErrorKind::UnicodeNotAllowed));
        }

        let mut buf = [0u8; 4];
        let i = unicode::encode_utf8(c, &mut buf).unwrap();
        let bytes = &buf[0..i];
        assert!(!bytes.is_empty());

        Ok(if bytes.len() == 1 {
            Hir::Literal(bytes[0])
        } else {
            Hir::concat(bytes.iter().cloned().map(Hir::Literal).collect())
        })
    }

    fn hir_from_char_case_insensitive(
        &self,
        span: Span,
        c: char,
    ) -> Result<Hir> {
        if !self.flags().unicode() && c.len_utf8() > 1 {
            return Err(self.error(span, ErrorKind::UnicodeNotAllowed));
        }
        // If case folding won't do anything, then don't bother trying.
        if !unicode::contains_simple_case_mapping(c, c) {
            return self.hir_from_char(span, c);
        }

        let mut cls = hir::ClassUnicode::new(vec![
            hir::ClassRangeUnicode::new(c, c),
        ]);
        cls.case_fold_simple();
        Ok(Hir::Class(hir::Class::Unicode(cls)))
    }

    fn hir_dot(&self, span: Span) -> Result<Hir> {
        let unicode = self.flags().unicode();
        Ok(if self.flags().dot_matches_new_line() {
            Hir::Class(if unicode {
                let ranges = vec![
                    hir::ClassRangeUnicode::new('\0', '\u{10FFFF}'),
                ];
                hir::Class::Unicode(hir::ClassUnicode::new(ranges))
            } else {
                if !self.trans().allow_invalid_utf8 {
                    return Err(self.error(span, ErrorKind::InvalidUtf8));
                }
                let ranges = vec![
                    hir::ClassRangeBytes::new(b'\0', b'\xFF'),
                ];
                hir::Class::Bytes(hir::ClassBytes::new(ranges))
            })
        } else {
            Hir::Class(if unicode {
                let ranges = vec![
                    hir::ClassRangeUnicode::new('\0', '\x09'),
                    hir::ClassRangeUnicode::new('\x0B', '\u{10FFFF}'),
                ];
                hir::Class::Unicode(hir::ClassUnicode::new(ranges))
            } else {
                if !self.trans().allow_invalid_utf8 {
                    return Err(self.error(span, ErrorKind::InvalidUtf8));
                }
                let ranges = vec![
                    hir::ClassRangeBytes::new(b'\0', b'\x09'),
                    hir::ClassRangeBytes::new(b'\x0B', b'\xFF'),
                ];
                hir::Class::Bytes(hir::ClassBytes::new(ranges))
            })
        })
    }

    fn hir_assertion(&self, asst: &ast::Assertion) -> Hir {
        let unicode = self.flags().unicode();
        let multi_line = self.flags().multi_line();
        match asst.kind {
            ast::AssertionKind::StartLine => {
                Hir::Anchor(if multi_line {
                    hir::Anchor::StartLine
                } else {
                    hir::Anchor::StartText
                })
            }
            ast::AssertionKind::EndLine => {
                Hir::Anchor(if multi_line {
                    hir::Anchor::EndLine
                } else {
                    hir::Anchor::EndText
                })
            }
            ast::AssertionKind::StartText => {
                Hir::Anchor(hir::Anchor::StartText)
            }
            ast::AssertionKind::EndText => {
                Hir::Anchor(hir::Anchor::EndText)
            }
            ast::AssertionKind::WordBoundary => {
                Hir::WordBoundary(if unicode {
                    hir::WordBoundary::Unicode
                } else {
                    hir::WordBoundary::Ascii
                })
            }
            ast::AssertionKind::NotWordBoundary => {
                Hir::WordBoundary(if unicode {
                    hir::WordBoundary::UnicodeNegate
                } else {
                    hir::WordBoundary::AsciiNegate
                })
            }
        }
    }

    fn hir_group(&self, group: &ast::Group, expr: Hir) -> Hir {
        let kind = match group.kind {
            ast::GroupKind::CaptureIndex(idx) => {
                hir::GroupKind::CaptureIndex(idx)
            }
            ast::GroupKind::CaptureName(ref capname) => {
                hir::GroupKind::CaptureName {
                    name: capname.name.clone(),
                    index: capname.index,
                }
            }
            ast::GroupKind::NonCapturing(_) => hir::GroupKind::NonCapturing,
        };
        Hir::Group(hir::Group {
            kind: kind,
            hir: Box::new(expr),
        })
    }

    fn hir_repetition(&self, rep: &ast::Repetition, expr: Hir) -> Hir {
        let kind = match rep.op.kind {
            ast::RepetitionKind::ZeroOrOne => hir::RepetitionKind::ZeroOrOne,
            ast::RepetitionKind::ZeroOrMore => hir::RepetitionKind::ZeroOrMore,
            ast::RepetitionKind::OneOrMore => hir::RepetitionKind::OneOrMore,
            ast::RepetitionKind::Range(ast::RepetitionRange::Exactly(m)) => {
                hir::RepetitionKind::Range(hir::RepetitionRange::Exactly(m))
            }
            ast::RepetitionKind::Range(ast::RepetitionRange::AtLeast(m)) => {
                hir::RepetitionKind::Range(hir::RepetitionRange::AtLeast(m))
            }
            ast::RepetitionKind::Range(ast::RepetitionRange::Bounded(m,n)) => {
                hir::RepetitionKind::Range(hir::RepetitionRange::Bounded(m, n))
            }
        };
        let greedy =
            if self.flags().swap_greed() {
                !rep.greedy
            } else {
                rep.greedy
            };
        Hir::Repetition(hir::Repetition {
            kind: kind,
            greedy: greedy,
            hir: Box::new(expr),
        })
    }

    fn hir_unicode_class(
        &self,
        ast_class: &ast::ClassUnicode,
    ) -> Result<hir::ClassUnicode> {
        use ast::ClassUnicodeKind::*;

        if !self.flags().unicode() {
            return Err(self.error(
                ast_class.span,
                ErrorKind::UnicodeNotAllowed,
            ));
        }
        let query = match ast_class.kind {
            OneLetter(name) => ClassQuery::OneLetter(name),
            Named(ref name) => ClassQuery::Binary(name),
            NamedValue { ref name, ref value, .. } => {
                ClassQuery::ByValue {
                    property_name: name,
                    property_value: value,
                }
            }
        };
        match unicode::class(query) {
            Ok(mut class) => {
                self.unicode_fold_and_negate(ast_class.negated, &mut class);
                Ok(class)
            }
            Err(unicode::Error::PropertyNotFound) => {
                Err(self.error(
                    ast_class.span,
                    ErrorKind::UnicodePropertyNotFound,
                ))
            }
            Err(unicode::Error::PropertyValueNotFound) => {
                Err(self.error(
                    ast_class.span,
                    ErrorKind::UnicodePropertyValueNotFound,
                ))
            }
        }
    }

    fn hir_perl_unicode_class(
        &self,
        ast_class: &ast::ClassPerl,
    ) -> hir::ClassUnicode {
        use ast::ClassPerlKind::*;
        use unicode_tables::perl_word::PERL_WORD;

        assert!(self.flags().unicode());
        let mut class = match ast_class.kind {
            Digit => {
                let query = ClassQuery::Binary("Decimal_Number");
                unicode::class(query).unwrap()
            }
            Space => {
                let query = ClassQuery::Binary("Whitespace");
                unicode::class(query).unwrap()
            }
            Word => unicode::hir_class(PERL_WORD),
        };
        // We needn't apply case folding here because the Perl Unicode classes
        // are already closed under Unicode simple case folding.
        if ast_class.negated {
            class.negate();
        }
        class
    }

    fn hir_perl_byte_class(
        &self,
        ast_class: &ast::ClassPerl,
    ) -> hir::ClassBytes {
        use ast::ClassPerlKind::*;

        assert!(!self.flags().unicode());
        let mut class = match ast_class.kind {
            Digit => hir_ascii_class_bytes(&ast::ClassAsciiKind::Digit),
            Space => hir_ascii_class_bytes(&ast::ClassAsciiKind::Space),
            Word => hir_ascii_class_bytes(&ast::ClassAsciiKind::Word),
        };
        // We needn't apply case folding here because the Perl ASCII classes
        // are already closed (under ASCII case folding).
        if ast_class.negated {
            class.negate();
        }
        class
    }

    fn unicode_fold_and_negate(
        &self,
        negated: bool,
        class: &mut hir::ClassUnicode,
    ) {
        // Note that we must apply case folding before negation!
        // Consider `(?i)[^x]`. If we applied negation field, then
        // the result would be the character class that matched any
        // Unicode scalar value.
        if self.flags().case_insensitive() {
            class.case_fold_simple();
        }
        if negated {
            class.negate();
        }
    }

    fn bytes_fold_and_negate(
        &self,
        ast: &ast::ClassBracketed,
        class: &mut hir::ClassBytes,
    ) -> Result<()> {
        // Note that we must apply case folding before negation!
        // Consider `(?i)[^x]`. If we applied negation field, then
        // the result would be the character class that matched any
        // Unicode scalar value.
        if self.flags().case_insensitive() {
            class.case_fold_simple();
        }
        if ast.negated {
            class.negate();
        }
        if !self.trans().allow_invalid_utf8 && !class.is_all_ascii() {
            return Err(self.error(ast.span, ErrorKind::InvalidUtf8));
        }
        Ok(())
    }

    /// Return a scalar byte value suitable for use as a literal in a byte
    /// character class.
    fn class_literal_byte(&self, ast: &ast::Literal) -> Result<u8> {
        match try!(self.literal_to_char(ast)) {
            Either::Left(ch) => {
                if ch <= 0x7F as char {
                    Ok(ch as u8)
                } else {
                    // We can't feasibly support Unicode in
                    // byte oriented classes. Byte classes don't
                    // do Unicode case folding.
                    Err(self.error(ast.span, ErrorKind::UnicodeNotAllowed))
                }
            }
            Either::Right(byte) => Ok(byte),
        }
    }
}

/// A translator's representation of a regular expression's flags at any given
/// moment in time.
///
/// Each flag can be in one of three states: absent, present but disabled or
/// present but enabled.
#[derive(Clone, Copy, Debug, Default)]
struct Flags {
    case_insensitive: Option<bool>,
    multi_line: Option<bool>,
    dot_matches_new_line: Option<bool>,
    swap_greed: Option<bool>,
    unicode: Option<bool>,
    // Note that `ignore_whitespace` is omitted here because it is handled
    // entirely in the parser.
}

impl Flags {
    fn from_ast(ast: &ast::Flags) -> Flags {
        let mut flags = Flags::default();
        let mut enable = true;
        for item in &ast.items {
            match item.kind {
                ast::FlagsItemKind::Negation => {
                    enable = false;
                }
                ast::FlagsItemKind::Flag(ast::Flag::CaseInsensitive) => {
                    flags.case_insensitive = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::MultiLine) => {
                    flags.multi_line = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::DotMatchesNewLine) => {
                    flags.dot_matches_new_line = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::SwapGreed) => {
                    flags.swap_greed = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::Unicode) => {
                    flags.unicode = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::IgnoreWhitespace) => {}
            }
        }
        flags
    }

    fn merge(&mut self, previous: &Flags) {
        if self.case_insensitive.is_none() {
            self.case_insensitive = previous.case_insensitive;
        }
        if self.multi_line.is_none() {
            self.multi_line = previous.multi_line;
        }
        if self.dot_matches_new_line.is_none() {
            self.dot_matches_new_line = previous.dot_matches_new_line;
        }
        if self.swap_greed.is_none() {
            self.swap_greed = previous.swap_greed;
        }
        if self.unicode.is_none() {
            self.unicode = previous.unicode;
        }
    }

    fn case_insensitive(&self) -> bool {
        self.case_insensitive.unwrap_or(false)
    }

    fn multi_line(&self) -> bool {
        self.multi_line.unwrap_or(false)
    }

    fn dot_matches_new_line(&self) -> bool {
        self.dot_matches_new_line.unwrap_or(false)
    }

    fn swap_greed(&self) -> bool {
        self.swap_greed.unwrap_or(false)
    }

    fn unicode(&self) -> bool {
        self.unicode.unwrap_or(true)
    }
}

fn hir_ascii_class_unicode(kind: &ast::ClassAsciiKind) -> hir::ClassUnicode {
    let ranges: Vec<_> = ascii_class(kind).iter().cloned().map(|(s, e)| {
        hir::ClassRangeUnicode::new(s, e)
    }).collect();
    hir::ClassUnicode::new(ranges)
}

fn hir_ascii_class_bytes(kind: &ast::ClassAsciiKind) -> hir::ClassBytes {
    let ranges: Vec<_> = ascii_class(kind).iter().cloned().map(|(s, e)| {
        hir::ClassRangeBytes::new(s as u8, e as u8)
    }).collect();
    hir::ClassBytes::new(ranges)
}

fn ascii_class(kind: &ast::ClassAsciiKind) -> &'static [(char, char)] {
    use ast::ClassAsciiKind::*;

    // TODO: Get rid of these consts, which appear necessary for older
    // versions of Rust.
    type T = &'static [(char, char)];
    match *kind {
        Alnum => {
            const X: T = &[('0', '9'), ('A', 'Z'), ('a', 'z')];
            X
        }
        Alpha => {
            const X: T = &[('A', 'Z'), ('a', 'z')];
            X
        }
        Ascii => {
            const X: T = &[('\x00', '\x7F')];
            X
        }
        Blank => {
            const X: T = &[(' ', '\t')];
            X
        }
        Cntrl => {
            const X: T = &[('\x00', '\x1F'), ('\x7F', '\x7F')];
            X
        }
        Digit => {
            const X: T = &[('0', '9')];
            X
        }
        Graph => {
            const X: T = &[('!', '~')];
            X
        }
        Lower => {
            const X: T = &[('a', 'z')];
            X
        }
        Print => {
            const X: T = &[(' ', '~')];
            X
        }
        Punct => {
            const X: T = &[('!', '/'), (':', '@'), ('[', '`'), ('{', '~')];
            X
        }
        Space => {
            const X: T = &[
                ('\t', '\t'), ('\n', '\n'), ('\x0B', '\x0B'), ('\x0C', '\x0C'),
                ('\r', '\r'), (' ', ' '),
            ];
            X
        }
        Upper => {
            const X: T = &[('A', 'Z')];
            X
        }
        Word => {
            const X: T = &[('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')];
            X
        }
        Xdigit => {
            const X: T = &[('0', '9'), ('A', 'F'), ('a', 'f')];
            X
        }
    }
}

#[cfg(test)]
mod tests {
    use ast::{Ast, Position, Span};
    use ast::parse::Parser;
    use hir::{self, Hir};
    use super::TranslatorBuilder;

    // We create these errors to compare with real hir::Errors in the tests.
    // We define equality between TestError and hir::Error to disregard the
    // pattern string in hir::Error, which is annoying to provide in tests.
    #[derive(Clone, Debug)]
    struct TestError {
        span: Span,
        kind: hir::ErrorKind,
    }

    impl PartialEq<hir::Error> for TestError {
        fn eq(&self, other: &hir::Error) -> bool {
            self.span == other.span && self.kind == other.kind
        }
    }

    impl PartialEq<TestError> for hir::Error {
        fn eq(&self, other: &TestError) -> bool {
            self.span == other.span && self.kind == other.kind
        }
    }

    fn parse(pattern: &str) -> Ast {
        Parser::new().parse(pattern).unwrap()
    }

    fn t(pattern: &str) -> Hir {
        TranslatorBuilder::new()
            .allow_invalid_utf8(false)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap()
    }

    fn t_err(pattern: &str) -> hir::Error {
        TranslatorBuilder::new()
            .allow_invalid_utf8(false)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap_err()
    }

    fn t_bytes(pattern: &str) -> Hir {
        TranslatorBuilder::new()
            .allow_invalid_utf8(true)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap()
    }

    fn hir_lit(s: &str) -> Hir {
        match s.len() {
            0 => Hir::Empty,
            1 => Hir::Literal(s.as_bytes()[0]),
            _ => {
                let lits = s
                    .as_bytes()
                    .iter()
                    .cloned()
                    .map(Hir::Literal)
                    .collect();
                Hir::concat(lits)
            }
        }
    }

    fn hir_group(i: u32, expr: Hir)  -> Hir {
        Hir::Group(hir::Group {
            kind: hir::GroupKind::CaptureIndex(i),
            hir: Box::new(expr),
        })
    }

    fn hir_group_name(i: u32, name: &str, expr: Hir)  -> Hir {
        Hir::Group(hir::Group {
            kind: hir::GroupKind::CaptureName {
                name: name.to_string(),
                index: i,
            },
            hir: Box::new(expr),
        })
    }

    fn hir_group_nocap(expr: Hir)  -> Hir {
        Hir::Group(hir::Group {
            kind: hir::GroupKind::NonCapturing,
            hir: Box::new(expr),
        })
    }

    fn hir_quest(greedy: bool, expr: Hir) -> Hir {
        Hir::Repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrOne,
            greedy: greedy,
            hir: Box::new(expr),
        })
    }

    fn hir_star(greedy: bool, expr: Hir) -> Hir {
        Hir::Repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: greedy,
            hir: Box::new(expr),
        })
    }

    fn hir_plus(greedy: bool, expr: Hir) -> Hir {
        Hir::Repetition(hir::Repetition {
            kind: hir::RepetitionKind::OneOrMore,
            greedy: greedy,
            hir: Box::new(expr),
        })
    }

    fn hir_range(greedy: bool, range: hir::RepetitionRange, expr: Hir) -> Hir {
        Hir::Repetition(hir::Repetition {
            kind: hir::RepetitionKind::Range(range),
            greedy: greedy,
            hir: Box::new(expr),
        })
    }

    fn hir_alt(alts: Vec<Hir>) -> Hir {
        Hir::Alternation(alts)
    }

    fn hir_cat(exprs: Vec<Hir>) -> Hir {
        Hir::Concat(exprs)
    }

    fn hir_uclass(ranges: &[(char, char)]) -> Hir {
        let ranges: Vec<hir::ClassRangeUnicode> = ranges
            .iter()
            .map(|&(s, e)| hir::ClassRangeUnicode::new(s, e))
            .collect();
        Hir::Class(hir::Class::Unicode(hir::ClassUnicode::new(ranges)))
    }

    fn hir_bclass(ranges: &[(u8, u8)]) -> Hir {
        let ranges: Vec<hir::ClassRangeBytes> = ranges
            .iter()
            .map(|&(s, e)| hir::ClassRangeBytes::new(s, e))
            .collect();
        Hir::Class(hir::Class::Bytes(hir::ClassBytes::new(ranges)))
    }

    fn hir_anchor(anchor: hir::Anchor) -> Hir {
        Hir::Anchor(anchor)
    }

    fn hir_word(wb: hir::WordBoundary) -> Hir {
        Hir::WordBoundary(wb)
    }

    #[test]
    fn empty() {
        assert_eq!(t(""), Hir::Empty);
        assert_eq!(t("()"), hir_group(1, Hir::Empty));
        assert_eq!(t("|"), hir_alt(vec![Hir::Empty, Hir::Empty]));
        assert_eq!(t("()|()"), hir_alt(vec![
            hir_group(1, Hir::Empty),
            hir_group(2, Hir::Empty),
        ]));
        assert_eq!(t("(|b)"), hir_group(1, hir_alt(vec![
            Hir::Empty,
            hir_lit("b"),
        ])));
        assert_eq!(t("(a|)"), hir_group(1, hir_alt(vec![
            hir_lit("a"),
            Hir::Empty,
        ])));
        assert_eq!(t("(a||c)"), hir_group(1, hir_alt(vec![
            hir_lit("a"),
            Hir::Empty,
            hir_lit("c"),
        ])));
        assert_eq!(t("(||)"), hir_group(1, hir_alt(vec![
            Hir::Empty,
            Hir::Empty,
            Hir::Empty,
        ])));
    }

    #[test]
    fn literal() {
        assert_eq!(t("a"), hir_lit("a"));
        assert_eq!(t("(?-u)a"), hir_lit("a"));
        assert_eq!(t("☃"), hir_lit("☃"));
        assert_eq!(t("abcd"), hir_lit("abcd"));

        assert_eq!(t_err("(?-u)☃"), TestError {
            kind: hir::ErrorKind::UnicodeNotAllowed,
            span: Span::new(
                Position::new(5, 1, 6),
                Position::new(8, 1, 7),
            ),
        });
    }

    #[test]
    fn literal_case_insensitive() {
        assert_eq!(t("(?i)a"), hir_uclass(&[
            ('A', 'A'), ('a', 'a'),
        ]));
        assert_eq!(t("(?i:a)"), hir_group_nocap(hir_uclass(&[
            ('A', 'A'), ('a', 'a')],
        )));
        assert_eq!(t("a(?i)a(?-i)a"), hir_cat(vec![
            hir_lit("a"),
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_lit("a"),
        ]));
        assert_eq!(t("(?i)ab@c"), hir_cat(vec![
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_uclass(&[('B', 'B'), ('b', 'b')]),
            hir_lit("@"),
            hir_uclass(&[('C', 'C'), ('c', 'c')]),
        ]));
        assert_eq!(t("(?i)β"), hir_uclass(&[
            ('Β', 'Β'), ('β', 'β'), ('ϐ', 'ϐ'),
        ]));

        assert_eq!(t("(?i-u)a"), hir_uclass(&[
            ('A', 'A'), ('a', 'a'),
        ]));
        assert_eq!(t("(?-u)a(?i)a(?-i)a"), hir_cat(vec![
            hir_lit("a"),
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_lit("a"),
        ]));
        assert_eq!(t("(?i-u)ab@c"), hir_cat(vec![
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_uclass(&[('B', 'B'), ('b', 'b')]),
            hir_lit("@"),
            hir_uclass(&[('C', 'C'), ('c', 'c')]),
        ]));

        assert_eq!(t_err("(?i-u)β"), TestError {
            kind: hir::ErrorKind::UnicodeNotAllowed,
            span: Span::new(
                Position::new(6, 1, 7),
                Position::new(8, 1, 8),
            ),
        });
    }

    #[test]
    fn dot() {
        assert_eq!(t("."), hir_uclass(&[
            ('\0', '\t'),
            ('\x0B', '\u{10FFFF}'),
        ]));
        assert_eq!(t("(?s)."), hir_uclass(&[
            ('\0', '\u{10FFFF}'),
        ]));
        assert_eq!(t_bytes("(?-u)."), hir_bclass(&[
            (b'\0', b'\t'),
            (b'\x0B', b'\xFF'),
        ]));
        assert_eq!(t_bytes("(?s-u)."), hir_bclass(&[
            (b'\0', b'\xFF'),
        ]));

        // If invalid UTF-8 isn't allowed, then non-Unicode `.` isn't allowed.
        assert_eq!(t_err("(?-u)."), TestError {
            kind: hir::ErrorKind::InvalidUtf8,
            span: Span::new(Position::new(5, 1, 6), Position::new(6, 1, 7)),
        });
        assert_eq!(t_err("(?s-u)."), TestError {
            kind: hir::ErrorKind::InvalidUtf8,
            span: Span::new(Position::new(6, 1, 7), Position::new(7, 1, 8)),
        });
    }

    #[test]
    fn assertions() {
        assert_eq!(t("^"), hir_anchor(hir::Anchor::StartText));
        assert_eq!(t("$"), hir_anchor(hir::Anchor::EndText));
        assert_eq!(t(r"\A"), hir_anchor(hir::Anchor::StartText));
        assert_eq!(t(r"\z"), hir_anchor(hir::Anchor::EndText));
        assert_eq!(t("(?m)^"), hir_anchor(hir::Anchor::StartLine));
        assert_eq!(t("(?m)$"), hir_anchor(hir::Anchor::EndLine));
        assert_eq!(t(r"(?m)\A"), hir_anchor(hir::Anchor::StartText));
        assert_eq!(t(r"(?m)\z"), hir_anchor(hir::Anchor::EndText));

        assert_eq!(t(r"\b"), hir_word(hir::WordBoundary::Unicode));
        assert_eq!(t(r"\B"), hir_word(hir::WordBoundary::UnicodeNegate));
        assert_eq!(t(r"(?-u)\b"), hir_word(hir::WordBoundary::Ascii));
        assert_eq!(t(r"(?-u)\B"), hir_word(hir::WordBoundary::AsciiNegate));
    }

    #[test]
    fn group() {
        assert_eq!(t("(a)"), hir_group(1, hir_lit("a")));
        assert_eq!(t("(a)(b)"), hir_cat(vec![
            hir_group(1, hir_lit("a")),
            hir_group(2, hir_lit("b")),
        ]));
        assert_eq!(t("(a)|(b)"), hir_alt(vec![
            hir_group(1, hir_lit("a")),
            hir_group(2, hir_lit("b")),
        ]));
        assert_eq!(t("(?P<foo>)"), hir_group_name(1, "foo", Hir::Empty));
        assert_eq!(t("(?P<foo>a)"), hir_group_name(1, "foo", hir_lit("a")));
        assert_eq!(t("(?P<foo>a)(?P<bar>b)"), hir_cat(vec![
            hir_group_name(1, "foo", hir_lit("a")),
            hir_group_name(2, "bar", hir_lit("b")),
        ]));
        assert_eq!(t("(?:)"), hir_group_nocap(Hir::Empty));
        assert_eq!(t("(?:a)"), hir_group_nocap(hir_lit("a")));
        assert_eq!(t("(?:a)(b)"), hir_cat(vec![
            hir_group_nocap(hir_lit("a")),
            hir_group(1, hir_lit("b")),
        ]));
        assert_eq!(t("(a)(?:b)(c)"), hir_cat(vec![
            hir_group(1, hir_lit("a")),
            hir_group_nocap(hir_lit("b")),
            hir_group(2, hir_lit("c")),
        ]));
        assert_eq!(t("(a)(?P<foo>b)(c)"), hir_cat(vec![
            hir_group(1, hir_lit("a")),
            hir_group_name(2, "foo", hir_lit("b")),
            hir_group(3, hir_lit("c")),
        ]));
    }

    #[test]
    fn flags() {
        assert_eq!(t("(?i:a)a"), hir_cat(vec![
            hir_group_nocap(hir_uclass(&[('A', 'A'), ('a', 'a')])),
            hir_lit("a"),
        ]));
        assert_eq!(t("(?i-u:a)β"), hir_cat(vec![
            hir_group_nocap(hir_uclass(&[('A', 'A'), ('a', 'a')])),
            hir_lit("β"),
        ]));
        assert_eq!(t("(?i)(?-i:a)a"), hir_cat(vec![
            hir_group_nocap(hir_lit("a")),
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
        ]));
        assert_eq!(t("(?im)a^"), hir_cat(vec![
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_anchor(hir::Anchor::StartLine),
        ]));
        assert_eq!(t("(?im)a^(?i-m)a^"), hir_cat(vec![
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_anchor(hir::Anchor::StartLine),
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
            hir_anchor(hir::Anchor::StartText),
        ]));
        assert_eq!(t("(?U)a*a*?(?-U)a*a*?"), hir_cat(vec![
            hir_star(false, hir_lit("a")),
            hir_star(true, hir_lit("a")),
            hir_star(true, hir_lit("a")),
            hir_star(false, hir_lit("a")),
        ]));
        assert_eq!(t("(?:a(?i)a)a"), hir_cat(vec![
            hir_group_nocap(hir_cat(vec![
                hir_lit("a"),
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
            ])),
            hir_lit("a"),
        ]));
        assert_eq!(t("(?i)(?:a(?-i)a)a"), hir_cat(vec![
            hir_group_nocap(hir_cat(vec![
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_lit("a"),
            ])),
            hir_uclass(&[('A', 'A'), ('a', 'a')]),
        ]));
    }

    #[test]
    fn escape() {
        assert_eq!(
            t(r"\\\.\+\*\?\(\)\|\[\]\{\}\^\$\#"),
            hir_lit(r"\.+*?()|[]{}^$#"),
        );
    }

    #[test]
    fn repetition() {
        assert_eq!(t("a?"), hir_quest(true, hir_lit("a")));
        assert_eq!(t("a*"), hir_star(true, hir_lit("a")));
        assert_eq!(t("a+"), hir_plus(true, hir_lit("a")));
        assert_eq!(t("a??"), hir_quest(false, hir_lit("a")));
        assert_eq!(t("a*?"), hir_star(false, hir_lit("a")));
        assert_eq!(t("a+?"), hir_plus(false, hir_lit("a")));

        assert_eq!(
            t("a{1}"),
            hir_range(
                true,
                hir::RepetitionRange::Exactly(1),
                hir_lit("a"),
            ));
        assert_eq!(
            t("a{1,}"),
            hir_range(
                true,
                hir::RepetitionRange::AtLeast(1),
                hir_lit("a"),
            ));
        assert_eq!(
            t("a{1,2}"),
            hir_range(
                true,
                hir::RepetitionRange::Bounded(1, 2),
                hir_lit("a"),
            ));
        assert_eq!(
            t("a{1}?"),
            hir_range(
                false,
                hir::RepetitionRange::Exactly(1),
                hir_lit("a"),
            ));
        assert_eq!(
            t("a{1,}?"),
            hir_range(
                false,
                hir::RepetitionRange::AtLeast(1),
                hir_lit("a"),
            ));
        assert_eq!(
            t("a{1,2}?"),
            hir_range(
                false,
                hir::RepetitionRange::Bounded(1, 2),
                hir_lit("a"),
            ));

        assert_eq!(t("ab?"), hir_cat(vec![
            hir_lit("a"),
            hir_quest(true, hir_lit("b")),
        ]));
        assert_eq!(t("(ab)?"), hir_quest(true, hir_group(1, hir_cat(vec![
            hir_lit("a"),
            hir_lit("b"),
        ]))));
        assert_eq!(t("a|b?"), hir_alt(vec![
            hir_lit("a"),
            hir_quest(true, hir_lit("b")),
        ]));
    }

    #[test]
    fn cat_alt() {
        assert_eq!(t("(ab)"), hir_group(1, hir_cat(vec![
            hir_lit("a"),
            hir_lit("b"),
        ])));
        assert_eq!(t("a|b"), hir_alt(vec![
            hir_lit("a"),
            hir_lit("b"),
        ]));
        assert_eq!(t("a|b|c"), hir_alt(vec![
            hir_lit("a"),
            hir_lit("b"),
            hir_lit("c"),
        ]));
        assert_eq!(t("ab|bc|cd"), hir_alt(vec![
            hir_lit("ab"),
            hir_lit("bc"),
            hir_lit("cd"),
        ]));
        assert_eq!(t("(a|b)"), hir_group(1, hir_alt(vec![
            hir_lit("a"),
            hir_lit("b"),
        ])));
        assert_eq!(t("(a|b|c)"), hir_group(1, hir_alt(vec![
            hir_lit("a"),
            hir_lit("b"),
            hir_lit("c"),
        ])));
        assert_eq!(t("(ab|bc|cd)"), hir_group(1, hir_alt(vec![
            hir_lit("ab"),
            hir_lit("bc"),
            hir_lit("cd"),
        ])));
        assert_eq!(t("(ab|(bc|(cd)))"), hir_group(1, hir_alt(vec![
            hir_lit("ab"),
            hir_group(2, hir_alt(vec![
                hir_lit("bc"),
                hir_group(3, hir_lit("cd")),
            ])),
        ])));
    }
}
