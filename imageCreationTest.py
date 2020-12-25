from PIL import Image, ImageDraw

img = Image.new('RGB', (640, 480), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((20, 20), "1", fill=(255, 0, 0))
img.save("Jon2", 'png')
img.close()
exit(0)