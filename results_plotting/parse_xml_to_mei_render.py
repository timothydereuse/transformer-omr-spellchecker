import music21 as m21
import os
import verovio

musicxml_name = (
    r"C:\Users\tim\Documents\transformer-midi-error-correction\test2.musicxml"
)

tk = verovio.toolkit()
tk.loadFile(musicxml_name)
tk.getPageCount()
svg_string = tk.renderToSVG(1)
tk.renderToSVGFile("page.svg", 1)
