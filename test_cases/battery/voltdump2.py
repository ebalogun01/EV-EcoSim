"""This file comprises the functions for parsing the voltages at every single timestep from the powerflow simulation
into a readable, clean csv"""

import csv
import glmptime as glmptime


def parse_voltages(path_prefix):
    """This function parses the voltages from gridLabD module into a readable csv
    Inputs: path_prefix - path defining folder in which to save the new voltages.csv file"""
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
                    Ar = float(row[1])
                    Ai = float(row[2])
                    Br = float(row[3])
                    Bi = float(row[4])
                    Cr = float(row[5])
                    Ci = float(row[6])
                    if f"{node}_Ar" not in nodes:
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
            writer.writerow(row)
