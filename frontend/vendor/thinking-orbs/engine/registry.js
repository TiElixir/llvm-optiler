// Ported from https://github.com/RareFormLabs/thinking-orbs (MIT, (c) 2026 Jakub Antalik): TypeScript types stripped, relative imports given .js extensions. See LICENSE.
// Mode key → frame painter. Kept separate from the presets so tree
// shaking can in principle drop unused modes in custom builds.

                                          
                                        
import { drawGlobe, drawRubik, drawWave } from './lattice.js';
import { drawMorph } from './morph.js';
import { drawOrbits } from './orbits.js';
import { drawRibbon } from './ribbon.js';

export const MODE_DRAWS                            = {
  orbits: drawOrbits,
  globe: drawGlobe,
  rubik: drawRubik,
  wave: drawWave,
  ribbon: drawRibbon,
  morph: drawMorph
};
