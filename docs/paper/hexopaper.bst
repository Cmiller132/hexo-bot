%% Minimal plain numeric BibTeX style for the paper's five fixed references.
ENTRY
  { author title journal volume year pages publisher school
    howpublished url }
  { }
  { }

FUNCTION {print.field}
{ duplicate$ empty$ { pop$ } { write$ } if$ }

FUNCTION {begin.bib}
{ preamble$ empty$ { skip$ } { preamble$ write$ newline$ } if$
  "\begin{thebibliography}{99}" write$ newline$ }

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

FUNCTION {begin.item}
{ newline$
  "\bibitem{" write$ cite$ write$ "}" write$ newline$ }

FUNCTION {article}
{ begin.item
  author print.field
  ". " write$ title print.field
  ". \emph{" write$ journal print.field "}" write$
  volume empty$ { skip$ } { ", " write$ volume print.field } if$
  pages empty$ { skip$ } { ", pp. " write$ pages print.field } if$
  year empty$ { skip$ } { " (" write$ year print.field ")" write$ } if$
  "." write$ newline$ }

FUNCTION {book}
{ begin.item
  author print.field
  ". \emph{" write$ title print.field "}. " write$
  publisher print.field
  year empty$ { skip$ } { ", " write$ year print.field } if$
  "." write$ newline$ }

FUNCTION {phdthesis}
{ begin.item
  author print.field
  ". \emph{" write$ title print.field
  "}. PhD thesis, " write$ school print.field
  year empty$ { skip$ } { ", " write$ year print.field } if$
  "." write$ newline$ }

FUNCTION {misc}
{ begin.item
  author print.field
  ". \emph{" write$ title print.field "}. " write$
  howpublished print.field
  year empty$ { skip$ } { ", " write$ year print.field } if$
  url empty$ { skip$ }
    { ". \url{" write$ url print.field "}" write$ } if$
  "." write$ newline$ }

FUNCTION {default.type} { misc }

READ
EXECUTE {begin.bib}
ITERATE {call.type$}
EXECUTE {end.bib}
