import csv
import math
import sys
import os

import matplotlib.pyplot as plt


def read_output(file_name):
  ground_truth = []
  with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
      data = list(map(float, row))
      ground_truth.append(data)
  return ground_truth


def read_sensor_data(file_name):
  lidar = []
  radar = []
  with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
      label = row[0]
      data = list(map(float, row[1:]))
      if label == 'LIDAR':
        lidar.append(data)
      elif label == 'RADAR':
        radar.append(data)
      else:
        print('bad label in sensor data: ', label)
  return lidar, radar


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("python plot.py sensor_data.csv output_data.csv")
    exit(0)

  lidar, radar = read_sensor_data(sys.argv[1])
  print("Read {} lidar points and {} radar points from {}".format(len(lidar), len(radar), sys.argv[1]))
  output_data = read_output(sys.argv[2])
  print("Read {} lines from {}".format(len(output_data), sys.argv[2]))
  
  plt.figure(0)
  # plot lidar
  plt.plot([d[1] for d in lidar],
            [d[2] for d in lidar],
            'g*')
  # plot radar
  plt.plot([d[1] * math.cos(d[2]) for d in radar],
            [d[1] * math.sin(d[2]) for d in radar],
            'bx')
  # plot output
  plt.plot([d[1] for d in output_data],
            [d[2] for d in output_data],
            'r-')
  plt.legend(['lidar', 'radar', 'output'])
  plt.show()
