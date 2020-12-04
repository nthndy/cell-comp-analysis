run("Image Sequence...", "open=/home/nathan/data/SHARC/fucci/fucci1_171201/glimpses/cell_ID_13/gfp/cell_ID_13_gfp_t0.tif sort use");
run("Image Sequence...", "open=/home/nathan/data/SHARC/fucci/fucci1_171201/glimpses/cell_ID_13/rfp/cell_ID_13_rfp_t0.tif sort use");
run("Merge Channels...", "c1=rfp c2=gfp create");
saveAs("Tiff", "/home/nathan/data/SHARC/fucci/fucci1_171201/glimpses/cell_ID_13/Composite_13.tif");
run("AVI... ", "compression=JPEG frame=50 save=/home/nathan/data/SHARC/fucci/fucci1_171201/glimpses/cell_ID_13/Composite_13.avi");