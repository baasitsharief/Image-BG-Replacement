import cv2
import numpy as np
import argparse
import os

# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.rcParams["figure.figsize"] = (10, 10)

parser = argparse.ArgumentParser("Performs median cut color quantization on RGB image")
parser.add_argument("-D","--depth", type=int, help="Bit-Depth for quantization")
parser.add_argument("-I","--input", type=str, help="Path to Input Image")
parser.add_argument("-O", "--output", type=str, help="Path/Filename to store output image")

# def show(image):
#   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# show(img)

def quantize(img, bit_depth):

  #median cut clustering
  final_groups = list()
  pixels = list()
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pixel = np.zeros((5,))
      pixel[0] = img[i,j,0]
      pixel[1] = img[i,j,1]
      pixel[2] = img[i,j,2]
      pixel[3] = i
      pixel[4] = j 
      pixels.append(pixel)
  final_groups.append(pixels.copy())
  del pixels
  for _ in range(bit_depth):
    temp = final_groups.copy()
    final_groups.clear()
    for pixels in temp:
      pixels = np.array(pixels)
      max_range_index = np.argmax([np.max(pixels[:, ch])-np.min(pixels[:, ch]) for ch in range(img.shape[2])])
      pixels = pixels[pixels[:,max_range_index].argsort()]
      groups = [pixels[:(len(pixels)+1)//2], pixels[len(pixels)//2:]]
      for gp in groups:
        final_groups.append(gp)
  
  # #numpy array to store resulting image
  new_img = np.zeros(img.shape)

  for group in final_groups:
    group = np.array(group)
    b_mean = np.mean(group[:,0]).astype(np.uint8)
    g_mean = np.mean(group[:,1]).astype(np.uint8)
    r_mean = np.mean(group[:,2]).astype(np.uint8)
    for pixel in group:
      new_img[int(pixel[3]), int(pixel[4]), :] = np.array([b_mean, g_mean, r_mean])
  new_img = new_img.astype(np.uint8)
  
  return new_img

def main():
  args = parser.parse_args()
  bit_depth = args.depth
  input_path = args.input
  output_path = args.output
  ext = input_path.split('.')[-1]
  filename = os.path.split(input_path)[-1].split('.')[0]

  img = cv2.imread(input_path)
  q_img = quantize(img, bit_depth)
  if(output_path!=None):
    cv2.imwrite(output_path, q_img)
  else:
    cv2.imwrite(f"{filename}_{bit_depth}_bit.{ext}", q_img)

if __name__ == '__main__':
  main()