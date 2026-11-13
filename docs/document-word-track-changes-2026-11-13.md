# Word tracked text changes

New DOCX analyses show **Word tracked text changes** under **Media
Evidence**. The check reads inline insertions and deletions already stored
in top-level body paragraphs in the submitted Word file. It needs one
document. It can show a change that spans multiple sentences even when the
statistical writing checks only measure short contexts. [WordprocessingML inserted runs](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.insertedrun) · [Deleted runs](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.deletedrun)

Each retained excerpt gives its one-based body paragraph number and the
zero-based positions of the paragraph and revision element in the main
Word XML part, usually `word/document.xml`. Positions count elements.
The Standard PDF and JSON use the same record. The
JSON also records a hash of each displayed change's full text, including
characters beyond the excerpt. The signed package commits to the review
record and the original source. With the original DOCX, the verifier
reopens the XML and checks the recorded positions and text.

The check retains at most 24 excerpts of 240 characters each. It reports
the number of extra measured marks and excludes nested revisions, tables,
tracked paragraph marks, moves and formatting from its measured scope.
Authors and timestamps on revision marks are excluded. Redacted PDFs
withhold this section, including deleted text. The ordinary writing
checks continue to read their existing extracted input and disclose that
tracked wording may be omitted. A file may have had changes that were
accepted, removed, or made without Track Changes. Embedded marks do not
establish authorship or whether AI was involved.
[Word revision and paragraph mark semantics](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-accept-all-revisions-in-a-word-processing-document)

Two synthetic Word uploads went through the local Full worker with
simulated $9.99 checkout. The tracked file returned one insertion and one
deletion. The clean control completed with no matching marks. The API,
JSON, Standard PDFs and signed packages agreed, and local Chrome showed
the expected excerpts and exact XML positions. One statistical check
was skipped for its memory budget during this smoke test. The embedded
revision check runs on the Word source and adds zero confidence weight.
[Local verification receipt](evaluation-panels/document-word-track-changes-local-2026-11-13.json)
