import csv
import glmptime as glmptime


def parse_voltages(path_prefix):
    data = {}
    nodes = ["Timestamp"]
    lastnodes = []
    timestamp = None
    timezone = "UTC"
    with open(f'{path_prefix}volt_dump.csv', 'r') as dumpfile:
        print("Reading volt_dump...")
        reader = csv.reader(dumpfile)
        for row in reader:
            if row[0].startswith("#"):
                tpos = row[0].find(" at ")
                if tpos > 0:
                    timestamp = row[0][tpos + 4:tpos + 27]
                    timestamp = glmptime.glmptime(timestamp)
                    # if "00:00:00" in str(timestamp):
                    #	print(timestamp)
                    data[timestamp] = []
                    timezone = row[0][tpos + 24:tpos + 27]
                header = []
            elif not header:
                header = row
                assert (header == ['node_name', 'voltA_real', 'voltA_imag', 'voltB_real', 'voltB_imag', 'voltC_real',
                                   'voltC_imag'])
                if lastnodes:
                    assert (lastnodes == nodes)
                elif nodes:
                    lastnodes = nodes
            else:
                try:
                    node = row[0]
                    # A = complex(float(row[1]),float(row[2]))
                    Ar = float(row[1])
                    Ai = float(row[2])
                    # B = complex(float(row[3]),float(row[4]))
                    Br = float(row[3])
                    Bi = float(row[4])
                    # C = complex(float(row[5]),float(row[6]))
                    Cr = float(row[5])
                    Ci = float(row[6])
                    if f"{node}_Ar" not in nodes:
                        # nodes.extend([node+"_A",node+"_B",node+"_C"])
                        nodes.extend(
                            [
                                f"{node}_Ar",
                                f"{node}_Ai",
                                f"{node}_Br",
                                f"{node}_Bi",
                                f"{node}_Cr",
                                f"{node}_Ci",
                            ]
                        )
                    # data[timestamp].extend([A,B,C])
                    data[timestamp].extend([Ar, Ai, Br, Bi, Cr, Ci])

                except:
                    print(f"ERROR: ignored row '{row}'")

    with open(f'{path_prefix}voltages.csv', 'w') as voltages:
        print("Writing voltages...")
        writer = csv.writer(voltages)
        writer.writerow(nodes)
        for key in sorted(data.keys()):
            row = [key.strftime("%Y-%m-%dT%H:%M:%S%z")]
            row.extend("{0:.6f}".format(value) for value in data[key])
            # row.append("%g%+gd" % (value.real,value.imag))
            writer.writerow(row)
