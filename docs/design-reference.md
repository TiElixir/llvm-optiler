# Design reference: segfault.compilertech.org

Measured 2026-11-07 with getComputedStyle in Chrome, plus the site's own `:root` custom properties and CSS rules. The browser viewport was 1908 px wide (the window could not be resized to 1440 px). Section and card sizes are `clamp()` values that already hit their maximum below 1440 px, so they are identical at 1440 px. Tokens live in `frontend/src/styles/tokens.css`.

## Measured values

| Item | Value | Element / source |
|---|---|---|
| Page background | `#141110` | `html`, `body` (`--color-ink`) |
| Raised panel | `#191614` | "Fully online" bar, diagram cell (`--color-ink-panel`) |
| Footer | `#0f0d0c` | `footer` (`--color-ink-deep`) |
| Rule | `#2a2522`, 1px solid | cell, section, header, nav button borders (`--color-rule`) |
| Soft rule | `#201c1a` | faint vertical grid lines (`--color-rule-soft`) |
| Foreground | `#ede7e0` | body text, headings (`--color-bone`) |
| Body / muted | `#9a938c` | `p`, kickers (`--color-muted`) |
| Dim / faint | `#5c5651` | index numbers ("001"), diagram strokes (`--color-faint`); 2.6:1 on bg |
| Accent | `#4e03ff` | primary button, "Winners" label, FINALE diagram box, focus ring, `::selection` |
| Accent hover | `#5f18ff` | primary button on hover |
| Accent text | `#8a5cff` | violet text in the token field (`--color-accent-text`) |
| Text on accent | `#ede7e0` | measured on the button, not white |
| Error / ok | `#ff7a66` / `#6ee7a0` | site `--color-err` / `--color-ok` |
| Podium colors | `#d9b45b`, `#bfc4c9`, `#c0804a` | 1 px bars in the prizes section |
| Theme | dark only | `color-scheme: dark`, no `prefers-color-scheme` rule, no toggle in header (only Finalists link and menu button) |
| Base text | 16px / 24px, weight 500 | `body` |
| Body paragraph | Chalet London 15px / 23.25px (1.55), 500, `#9a938c` | `p` in cells, hero, sections |
| Secondary text | 14px / 20px, `#9a938c` | finalists meta lines |
| Section heading | 60px / 60px, 500, -1.5px (-0.025em), `#ede7e0`, centered | `h2.h-section`, `clamp(2rem, 5vw, 3.75rem)` |
| Card heading | 24px / 26.4px, 500, -0.6px, `#ede7e0`, left | `h3.h-card`, `clamp(1.25rem, 1.7vw, 1.5rem)` |
| Hero tagline | 28px / 28px, 500, -0.7px, centered | `p` under wordmark |
| Hero wordmark | SVG image, 768 x 110 px (max-w 48rem) | `h1 > img segfault-wordmark.svg`; no hero font-size exists. `.h-hero` is defined (`min(clamp(2.75rem, 7.4vw, 6.5rem), 10.2vw - 3px)`) but no element uses it |
| Prize figure | Source Code Pro 72px / 72px, 500, -3.6px | "₹2,00,000" |
| Kicker (cell) | Source Code Pro 11px / 16.5px, 500, 0.14em (1.54px), uppercase, `#9a938c` | "ANYWHERE", "ENTRY", buttons, nav |
| Kicker index | same as kicker, `#5c5651` | "001" |
| Kicker small | Source Code Pro 10px / 15px, 1.6px (0.16em), uppercase, `#9a938c` | "THE FORMAT", "WINNERS" |
| Tag row | Source Code Pro 10px, 1.2px, uppercase; `#9a938c` or `#5c5651` | repository/track tags |
| Date mono | Source Code Pro 13px / 19.5px, 0.52px | "Aug 1" |
| Bento grid | 3 columns of 394px, gap 0, 1px top border, cells divided by 1px left rules | 1182px grid inside a 1184px max-width column |
| Cell padding | 36px | cell content wrapper |
| Cell inner spacing | kicker/index row gap 16px; h3 margin-top 28px; paragraph margin-bottom 8px | Entry/Anywhere cells |
| Section padding | 96px 24px | heading wrapper |
| Cell background / hover | transparent, no hover change (0s transition) | hovered the "Entry" cell and inspected `:hover` ancestors |
| Header | sticky, 65px, `oklab(0.18 ...)` ~ `#141110` at 0.95 alpha, 1px bottom rule | `header` |
| Button | 40px high, padding 12px 20px, radius 0, mono 11px / 0.14em uppercase | "Finalists", "See tracks" |
| Secondary button | 1px frame `#2a2522` around a `#141110` inner span; hover frame goes to `#9a938c` over 0.2s | "See tracks" |
| Primary button | solid `#4e03ff`, hover `#5f18ff` | "Finalists" |
| Label pill | `#4e03ff` fill, 6px 12px, radius 0, Chalet London 16px | "Winners" |
| Menu button | 40 x 40, 1px `#2a2522`, radius 0 | header |
| Radius | 0px everywhere measured | cells, buttons, labels |
| Focus ring | 2px solid `#4e03ff`, offset 3px | `:focus-visible` |
| Selection | background `#4e03ff`, text `#ede7e0` | `::selection` |
| Diagram strokes | 1px `#5c5651`; one dashed rect (3px, 4px dash); animated path `#9a938c`, dasharray 12 88; connector paths `#5c5651` and `#2a2522` | "Fully online" SVG, 304 x 329 |
| Diagram accent box | `#4e03ff` fill, text `#ede7e0`, Source Code Pro 7.5px, 0.5px tracking | "FINALE · IISc" |
| Diagram labels | Source Code Pro 7.5px, fill `#5c5651` | REGISTER, TRACKS, SUBMIT, REVIEW |
| Token field | SVG at opacity 0.16, text at opacity 0.55 (about 0.088 effective), Source Code Pro 6.5px, 0.42px tracking, `#ede7e0` with some `#8a5cff` | footer-area background |

## Layout and fonts

The page is one centered column (max 1184px) framed by 1px vertical rules on a near-black warm background. Sections are hairline-bordered bento grids with no gaps and no rounded corners, and each cell is a mono uppercase kicker plus a faint index number over a tight-tracked sans heading and muted paragraph. Buttons are square, with the secondary one drawn as a 1px frame that brightens on hover; the violet accent shows up on a few solid fills and the focus ring. Diagrams are 1px-stroke SVG line drawings in the dim color with dashed boxes and one solid violet box, and a faint field of mono glyph noise sits behind some sections.

Fonts: body and headings use "chaletLondon" (file `ChaletLondonNineteenSixty`, TTF, weight 500); a "chaletNewYork" face (`ChaletNewYorkNineteenSixty`) is declared but never loaded or used. Mono labels use Source Code Pro (variable woff2, weight range 200-900, hashed file under `/_next/static/media/`). All font files and the one stylesheet come from the page's own host `segfault.compilertech.org` (Next.js `next/font`, no fonts.googleapis.com); no font was downloaded for this reference. Chalet is House Industries' commercial family, so it is not shipped here and Geist is the stand-in; Source Code Pro is OFL and can be self-hosted. Every text element measured uses weight 500, and the site loads no other weight.

## Not verified

- Mobile and 1440px-wide layouts: the browser window would not resize (viewport stayed 1908px). 1440px values follow from the `clamp()` formulas, not from a render.
- Cell hover: no background change was found; other hover effects (links, nav) were not exercised.
- A zero-size `#88c9f7` element with 2px border and 3px radius exists in the DOM; its purpose was not identified.
- The `--changed` and heat-2..4 steps in `tokens.css` are design choices, not measurements; heat-5 (0.55) is an estimate.
