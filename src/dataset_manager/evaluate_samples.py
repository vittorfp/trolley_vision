import cv2

print("\n\n\tGerador de classificoes\n\n\t\tDigite 't' para as fotos que tiverem o trolley\n\t\tDigite 'f' para as fotos NAO tiverem o trolley\n\t\tDigite 'q' para sair\n\n\tA classificao comeca de onde voce parou na ultima vez")
fh = open("labels.txt", "r") 
num = len(fh.readlines())
fh.close()
print("Iniciando da imagem " + str(num))

f = open("labels.txt", 'a')

for i in range(1008,271040):
	img = cv2.imread('../img/dataset2/img' + str(i) + '.jpg' )
	img = cv2.GaussianBlur(img,(5,5),0)
	cv2.imshow("Imagem", cv2.Canny(img, 30,45))
	key = cv2.waitKey(0)

	while not(key == 113 or key == 116 or key == 102):
		print("Opcao invalida")
		key = cv2.waitKey(0)
	
	# q	
	if(key == 113):
		break

	# 1
	if(key == 116):
		print("Com trolley")
		value = '1\n'
	# 0
	if(key == 102):
		print("Sem trolley")
		value = '0\n'

	f.write(value)


f.close()
