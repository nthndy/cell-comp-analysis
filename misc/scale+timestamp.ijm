open("/home/nathan/data/SHARC/fucci/fucci1_171201/glimpses/cell_ID_441/Cell_ID_441_rgb.tif");
run("Label...", "format=00:00 starting=145 interval=4 x=5 y=20 font=18 text=[] range=1-938");
run("Scale Bar...", "width=25 height=4 font=14 color=White background=None location=[Lower Right] overlay label");
run("Save");
