cap = cv2.VideoCapture(video_file)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

check , img = cap.read()
cv2.imwrite("/content/dummy_image" + ".jpg", img)

#Opening Image
im = Image.open("dummy_image.jpg")

print('Total Frame Count:', length )
print("Dummy_image of size {}".format(im.size))
print("="*50)
im
