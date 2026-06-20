import serial, csv

ser = serial.Serial('COM3', 115200) # Replace with your port
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    while True:
        line = ser.readline().decode().strip()
        if line.lower() == 'stop':
            break
        writer.writerow(line.split(','))