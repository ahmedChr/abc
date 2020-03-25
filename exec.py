import os
from pdf2image import convert_from_path
import tensorflow as tf
import time
import numpy as np
import json

import sys
import classificateur.scripts.label_image as lb
import os, shutil
import argparse

directory = 'converted'  
pages = convert_from_path("classificateur/pdfs/1.pdf", dpi=200)

def classify(image, graph):
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "classificateur/tf_files/retrained_graph.pb"
  label_file = "classificateur/tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  
  model_file = graph
  file_name = image
  
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = lb.load_graph(model_file)
  t = lb.read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = lb.load_labels(label_file)

  template = "{}"
  
  return labels[list(results).index(max(results))]
  
if not os.path.exists(directory):
    os.makedirs(directory)

def convertImages():
    for idx,page in enumerate(pages):
        page.save('converted/page'+str(idx)+'.jpg', 'JPEG')


def main():
    convertImages()
    response = []
    # iterate through the names of contents of the folder
    for i in range(len(os.listdir(directory))):
        response.append({
            "page: " + str(i+1):
            classify("converted/"+os.listdir(directory)[i], "classificateur/tf_files/retrained_graph.pb")
        })
        
    
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    print(response)
    return json.dumps(response)

