import cv2

print("\n\n\tGerador de classificoes\n\n\t\tDigite 't' para as fotos que tiverem o trolley\n\t\tDigite 'f' para as fotos NAO tiverem o trolley\n\t\tDigite 'q' para sair\n\n\tA classificao comeca de onde voce parou na ultima vez")
fh = open("corrente_labels.txt", "r") 
num = len(fh.readlines())
fh.close()
print("Iniciando da imagem " + str(num))

f = open("corrente_labels.txt", 'a')

for i in range(1,3252):
	img = cv2.imread('../../img/dataset3/img' + str(i) + '.jpg' )
	#Simg = cv2.GaussianBlur(img,(5,5),0)
	cv2.imshow("Imagem", img)
	key = cv2.waitKey(0)

	while not(key == 113 or key == 101 or key == 109 or key == 105 or key == 97):
		print("Opcao invalida")
		key = cv2.waitKey(0)
		#print(key)
	
	
	# q	= 113
	if(key == 113):
		break

	# e = 101
	if(key == 101):
		print("elo")
		value = 'elo\n'

	# m = 109
	if(key == 109):
		print("malha")
		value = 'malha\n'

	# a = 97
	if(key == 97):
		print("arrastador")
		value = 'arrastador\n'

	# i = 105
	if(key == 105):
		print("indefinido")
		value = 'indefinido\n'



	f.write(value)


f.close()
