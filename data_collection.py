import serial, csv

ser = serial.Serial('COM3', 115200) # Replace with your port
with open('data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    while True:
        line = ser.readline().decode().strip()
        if line.lower() == 'stop':
            break
        wd = line.split(',')
        writer.writerow(wd)
        print(f"Data collected: {wd}")
ser.close()