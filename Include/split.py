# split csv file

chunk_size = 15000000

def write_chunk(part, lines):
    with open('G:/My Drive/Bitcoin/Chunk/x_chunk_'+ str(part) +'.csv', 'w') as f_out:
        f_out.write(header)
        f_out.writelines(lines)

with open("G:/My Drive/Bitcoin/Copy of clean_data.csv", "r") as f:
    count = 0
    header = f.readline()
    lines = []
    for line in f:
        count += 1
        lines.append(line)
        if count % chunk_size == 0:
            write_chunk(count // chunk_size, lines)
            lines = []
    # write remainder
    if len(lines) > 0:
        write_chunk((count // chunk_size) + 1, lines)
