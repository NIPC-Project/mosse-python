import lib_png

for i in range(1, 375):
    _, _, rows, _ = lib_png.Reader(filename=f"frames-gray-png/{i}.png").read()
    png_bytes = bytes(0)
    for row in rows:
        png_bytes += bytes(row)
    print(len(png_bytes))
    with open(f"frames-gray-bin/{i}.bin", "wb") as f:
        f.write(png_bytes)
