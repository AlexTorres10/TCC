"""
Created on Mon Mar 30 22:26:10 2020

@author: alextorres10
"""

import os #biblioteca necessária para acessar os arquivos
REM = ["_","M","F"] #caracteres que eu quero remover

for name in os.listdir("."): # para cada nome no diretório atual
    origname = name #salvo o nome original
    for i in REM: #para cada caractere que quero tirar
        name = name.replace(i,"")# substituo por nada
    name = name.replace(".BP",".BMP")
    # Como eram .BMP, fiz isso para manter o formato do arquivo
    print(name) #Para mostrar como o nome ficou
    os.rename(origname, name) 
    #função que renomeia o arquivo (ORIGINAL,NOVONOME)