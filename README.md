# srm_motor
Models a solid rocket motor grain regression through use of a level set method on an Eulerian grid. 

The level set solver uses a first-order upwind gradient method based on a paper on this exact problem: https://scholar.sun.ac.za/bitstream/handle/10019.1/86526/sullwald_grain_2014.pdf.

The zero-set perimeter and interior area are solved using the python [contours](https://pypi.org/project/contours/) library.
