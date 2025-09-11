#!/usr/bin/env python3.9.17
# -*- Coding: UTF-8 -*-

'''
Estimate CF from ASI
author: Vinicius Roggério da Rocha
e-mail: vinicius.rocha@inpe.br
version: 0.0.2
date: 2023-07-10, 2025-09-10
'''

import os
from glob import glob
from datetime import datetime, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import cv2
from skimage.filters import threshold_li

class Aux():

    @staticmethod
    def pc_paths(pc_name):
        '''
        Define caminhos de arquivos conforme computador
        '''
        if pc_name == 'matrix':
            # Notebook Vinicius
            #path_in = '/data1/tsiskyimage/'
            path_in = '/home/vinicius/Documentos/INPE_outros/Unifei_furnas/asi_16068/'
            #path_out = '/home/vinicius/Documentos/doutorado/out/'
            path_out = '/home/vinicius/Documentos/INPE_outros/out'
        elif pc_name == 'rocinante':
            # Workstation INPE
            path_in = '/media/data/tsiskyimage/'
            path_out = '/home/vinicius/doutorado/out/'
        elif pc_name == 'desktop-pedro':
            # Windows Subsytem Linux
            path_in = '/mnt/c/vinicius/'
            path_out = '/mnt/c/vinicius/out/'
        return path_in, path_out

    @staticmethod
    def list_directories(path):
        '''
        Create directorie's list from path
        '''
        dir_list = sorted(os.listdir(path))
        return dir_list

    @staticmethod
    def list_files(path, ext):
        '''
        Create file's list from path with extension ext
        '''
        files = sorted([f for f in glob(path + "/*." + ext, recursive=True)])
        return files

    @staticmethod
    def filter_filelist(lista_arquivos, padrao):
        '''
        Filtrar lista de arquivos/strings conforme padrão
        '''
        return [arquivo for arquivo in lista_arquivos if arquivo.endswith(padrao)]

    @staticmethod
    def create_df(col_names):
        '''
        Create dataframe with given column names
        '''
        df = pd.DataFrame(columns=col_names)
        return df

    @staticmethod
    def date_str(filename, filetype):
        '''
        Get timestamp string and convert format
        '''
        if filetype == 'tsi':
            date_str = filename.split('/')[-1].split('.')[-2]
        elif filetype == 'sonda':
            date_str = filename.split('/')[-1].split('.')[0].split('_')[0]
        date_dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
        date_ts = datetime.strftime(date_dt, '%Y-%m-%d %H:%M:%S')
        return date_str, date_dt, date_ts

    @staticmethod
    def row_to_df(df, new_row):
        '''
        Insert row into a dataframe
        '''
        df.loc[len(df)] = new_row
        return df
    
    @staticmethod
    def save_df(df, filename):
        '''
        Save DF into CSV
        '''
        df.to_csv(filename, index=False)

    @staticmethod
    def check_mask(values_dict, lim_inf, lim_sup):
        '''
        Check if mask is OK or BAD (outside a range)
        '''
        dif = values_dict.get(0) - values_dict.get(255)
        if (dif < lim_inf) or (dif > lim_sup):
            flag = 'mask_bad'
        else:
            flag = 'mask_ok'
        return flag


class Img_func():

    @staticmethod
    def read_img(file_in):
        '''
        Load image
        cv2.IMREAD_COLOR: 1 (color image) DEFAULT
        cv2.IMREAD_GRAYSCALE: 0 (gray scale)
        cv2.IMREAD_UNCHANGED: -1 (include alpha channel)
        '''
        img = cv2.imread(file_in)
        #img = cv2.imread(file_in, cv2.IMREAD_GRAYSCALE)
        return img

    @staticmethod
    def zenith_radius_naive(img):
        '''
        Get xc, yc and radius only based in image shape
        '''
        hh, ww = img.shape[:2]
        xc = ww // 2
        yc = hh // 2
        rc = min(xc,yc)
        # Retirar 15% do raio (corresponde visualmente ao ângulo zenital de 80°)
        rc = int(round(0.85*rc,0))
        return xc, yc, rc

    @staticmethod
    def zenith_radius(img):
        '''
        Detect big circle and return circles:
         a list with (xc,yc,r) from every circle found (must be just one)
        https://stackoverflow.com/questions/60637120/detect-circles-in-opencv
        https://medium.com/turing-talks/houghcircles-detec%C3%A7%C3%A3o-de-c%C3%ADrculos-em-imagens-com-opencv-e-python-2d229ad9d43b
        '''
        # Prepar image: grayscale and blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 25)
        # Initial parameters
        minDist = 100
        param1 = 30
        param2 = 50 #smaller value-> more false circles
        # Raio maximo: metade da largura da imagem
        # Raio minimo: metade do raio máximo
        maxRadius = int(round(img.shape[0]/2,0))
        minRadius = int(round(maxRadius*0.5))
        # Detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist,
         param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            # Transformar em valores inteiros
            circles = np.uint16(np.around(circles))
            # Plot circles at image
            #for i in circles[0,:]:
            #    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Show result for testing
            #cv2.imwrite(f'circ_test.png', img)
            # Se tiver vários resultados, selecionar somente o primeiro
            if circles.shape != (1, 1, 3):
                circles = circles[:,0:1,:]
            # Selecionar valores
            xc = circles[0][0][0]
            yc = circles[0][0][1]
            rc = circles[0][0][2]
            # Fixar tamanho do círculo
            rc = 244
            # Retirar 15% do raio (corresponde visualmente ao ângulo zenital de 80°)
            rc = int(round(0.85*rc,0))
            # Plot circle at image
            #cv2.circle(img, (xc, yc), rc, (0, 255, 0), 2)
        else:
            #print('Some problem to finding ASI circle')
            xc = yc = rc = 0
        return xc, yc, rc

    @classmethod
    def create_mask(self, image, filetype):
        '''
        Create circular mask using HoughCircles
        Create mask using MCE to remove interferences
        '''
        # Get center coordinates (xc,yc) and radius from ROI
        if filetype == 'tsi':
            xc, yc, rc = self.zenith_radius(image)
        elif filetype == 'sonda':
            # Para ASI sem espelho, usar método mais simples para calcular informações do círculo
            xc, yc, rc = self.zenith_radius_naive(image)
        # Se não encontrou círculo, imagem deve tá bugada
        # Deixando dif=0, não entra no critério de seleção de máscara
        if rc == 0:
            return 'circle_problem', {0: -99, 255: -99}, [xc, yc, rc]
        # Crie uma máscara binária do mesmo tamanho da imagem original
        mascara = np.zeros(image.shape[:2], dtype=np.uint8)
        # Desenhe o círculo branco na máscara
        cv2.circle(mascara, (xc, yc), rc, 255, -1)

        # Calcular o valor do limiar usando MCE (Li and Lee, 1993; Li and Tam, 1998)
        try:
            threshold_value = threshold_li(image)
        except Exception as e:
            return 'threshold_problem', {0: -99, 255: -99}, [xc, yc, rc]
        # Binarizar a imagem usando o threshold obtido
        binary_image = np.where(image > threshold_value, 255, 0).astype(np.uint8)
        # Converter pixeis verdes (sugrem da binarização nos 3 canais) para preto
        mascara_verde = np.all(binary_image == [0, 255, 0], axis=-1)
        binary_image[mascara_verde] = [0, 0, 0]
        # Aplicar a máscara de MCE sobre a máscara circular - após converter para grayscale
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        final_mask = cv2.bitwise_and(binary_image, mascara, mask=mascara)
        # Transformar o cinza em branco/válido
        final_mask = np.where((final_mask > 0) & (final_mask < 255), 255, final_mask)
        
        # (controle) Totais de pixels pretos/mascara e brancos/válidos
        unique, counts = np.unique(final_mask, return_counts=True)
        counts_mask = dict(zip(unique, counts))

        return final_mask, counts_mask, [xc, yc, rc]

    @staticmethod
    def crop_circle_center(image, mask, circ_info):
        '''
        Recorta uma imagem quadrada de acordo com as informações do círculo.
        
        Args:
            image (numpy.ndarray): Imagem original.
            mask (numpy.ndarray): Máscara correspondente à imagem.
            xc (int): Coordenada x do centro do círculo.
            yc (int): Coordenada y do centro do círculo.
            rc (int): Raio do círculo - da ASI, não da área reduzida de r=210.
            
        Returns:
            numpy.ndarray: Imagem recortada.
            numpy.ndarray: Máscara recortada correspondente.
        '''
        xc, yc, rc_zen = circ_info
        # Aumentar 15% do raio apenas para cortar imagem
        rc = int(round(1.15*rc_zen,0))
        # Calcula as coordenadas do canto superior esquerdo do quadrado a ser recortado
        x1 = int(xc) - int(rc)
        y1 = int(yc) - int(rc)
        # Calcula as coordenadas do canto inferior direito do quadrado a ser recortado
        x2 = int(xc) + int(rc)
        y2 = int(yc) + int(rc)
        # Substituir valores negativos por zero
        x1, x2, y1, y2 = [max(x, 0) for x in [x1, x2, y1, y2]]
        
        # Recorta a imagem e a máscara
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        # Redefinir as coordenadas do círculo após o recorte
        hh, ww = cropped_image.shape[:2]
        xc = ww // 2
        yc = hh // 2
        #rc = min(xc,yc)
        circ_info = [xc, yc, rc_zen]
        
        return cropped_image, cropped_mask, circ_info

    @staticmethod
    def find_yellow_lab(img, mask):
        '''
        Encontrar e contabilizar pixels amarelados da imagem LAB
        '''
        # Contabilizar número total de pixels considerados na máscara (NonZero)
        n_total = cv2.countNonZero(mask)
        # Converter a imagem para o modelo de cores LAB
        imagem_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Aplicar a máscara à imagem LAB
        imagem_lab_mask = cv2.bitwise_and(imagem_lab, imagem_lab, mask=mask)
        # Definir o intervalo de cor para amarelo no espaço LAB
        lower_yellow = np.array([150, 0, 140], dtype=np.uint8)
        upper_yellow = np.array([255, 255, 255], dtype=np.uint8)
        # Encontrar pixels amarelados na imagem
        mascara_amarela = cv2.inRange(imagem_lab_mask, lower_yellow, upper_yellow)
        n_yellow = cv2.countNonZero(mascara_amarela)
        p_yellow = int(round((n_yellow*100)/n_total,0))
        return p_yellow

    @staticmethod
    def change_temperature(img, path_out, date_str):
        '''
        Alterar temperatura da imagem
        '''
        # Constante da temperatura (negativo para tornar mais azul/frio)
        constante = -40
        # Converter a imagem para o espaço de cor LAB
        lab_imagem = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Separar os canais LAB
        l, a, b = cv2.split(lab_imagem)
        # Ajuste da temperatura de cor no canal b*
        b = b - abs(constante)
        # Limitar os valores de b* para estar no intervalo de 0 a 255
        b = np.clip(b, 0, 255)
        # Juntar os canais novamente
        nova_imagem_lab = cv2.merge((l, a, b))
        # Converter a nova imagem de LAB para BGR
        nova_imagem = cv2.cvtColor(nova_imagem_lab, cv2.COLOR_LAB2BGR)

        # Plotar imagem
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # axs[0].set_title('Original')
        # axs[0].axis('off')
        # axs[1].imshow(cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2RGB))
        # axs[1].set_title('Ajustada')
        # axs[1].axis('off')
        # # Salva a figura como um arquivo PNG
        # output_path = f'{path_out}/{date_str}_temperatura_ajustada.png'
        # plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        # plt.close()
        return nova_imagem

    @staticmethod
    def NBRR(image):
        '''
        Calcula normalized blue/red ratio (NBRR)
        '''
        # Divide os canais RGB
        blue_channel = image[:, :, 0].astype(float)
        red_channel = image[:, :, 2].astype(float)
        # "Note that the value of r is increased by one unit
        # if it equals zero to avoid dividing by zero"
        red_channel[red_channel == 0] = 1
        # Calcula a NBRR
        nbrr = (blue_channel - red_channel)/(blue_channel + red_channel)
        # Normaliza a NBRR para valores entre 0 e 1 - não precisa
        #nbrr_normalized = (nbrr - np.nanmin(nbrr)) / (np.nanmax(nbrr) - np.nanmin(nbrr))
        # Multiplica o array pelo valor máximo (0.5*255=127, p. ex) e 
        # converte para o tipo uint8 (pra ficar valor inteiro)
        nbrr_uint8 = (nbrr * 255).astype(np.uint8)
        return nbrr_uint8, nbrr

    @staticmethod
    def calc_entropia_htw(img, mask):
        '''
        Calcular entropia e HTW de imagem (aplicando máscara)
        '''
        # Aplicar máscara para deixar preto os pontos que não interessam
        img = cv2.bitwise_and(img, img, mask=mask)
        # Remover pontos pretos, pois não fazem parte do céu/nuvem
        img_without_zero = img[img != 0]

        # CÁLCULO DE ENTROPIA
        # Calcule o histograma
        histograma = cv2.calcHist([img_without_zero], [0], None, [256], [0, 256])
        # Normalize o histograma
        probabilidades = histograma / np.sum(histograma)
        # Calcule a entropia
        entropia = -np.sum(probabilidades * np.log2(probabilidades + np.finfo(float).eps))

        # CÁLCULO DE HTW
        # Encontre os pontos onde a área acumulada atinge 25% e 75% da área total
        cumsum = np.cumsum(probabilidades)
        x_25 = np.argwhere(cumsum >= 0.25)[0][0]
        x_75 = np.argwhere(cumsum >= 0.75)[0][0]
        # Calcule a HTW
        htw = abs(x_75 - x_25)

        return entropia, htw

    @staticmethod
    def homog(entropia, htw):
        '''
        Definir critério de homogeneidade
        e parâmetros para MCE adaptativo
        '''
        if entropia > 6.3 and htw > 42:
            criterio_homogeneidade = 'heterogeneo'
        else:
            criterio_homogeneidade = 'homogeneo'
        criterio_homogeneidade = 'homogeneo' # FORÇANDO
        if criterio_homogeneidade == 'heterogeneo':
            block_size = 651  # Tamanho do bloco
            constant = 10.0   # Constante de ajuste
        else:
            block_size = 51  # Tamanho do bloco
            constant = 50.0   # Constante de ajuste
        return criterio_homogeneidade, block_size, constant

    @staticmethod
    def cld_seg(img_grayscale, mask, block_size, constant):
        '''
        Segmentação usando Adaptive Threshold
        '''
        # No tutorial, diz pra aplicar filtro blur antes (tira pontinhos espalhados)
        #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
        #img_grayscale = cv2.medianBlur(img_grayscale,5)
        # Aplicar máscara
        #img_grayscale = cv2.bitwise_and(img_grayscale, img_grayscale, mask=mask)

        # Calcula o limiar adaptativo usando o método MCE
        segmented_image = cv2.adaptiveThreshold(img_grayscale, 255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY,
                                                block_size, constant)
        # Definir cinza para nuvem (0->127) e branco para céu (já está)
        segmented_image[segmented_image == 0] = 127
        # Aplicar máscara (preto como interferências)
        segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=mask)
        return segmented_image

    @staticmethod
    def calc_cf(segmented_image):
        '''
        Calcular fração de cobertura de nuvens
        - Número de pixels de nuvem: n_gray
        - Número de pixels de céu claro: n_white
        - Número de pixels de interferência: n_black
        - Fração de cobertura de nuvem: n_gray/(n_gray+n_white)*100
        '''
        # Contabilizar cada cor
        n_black = np.sum(segmented_image == 0)
        n_gray = np.sum(segmented_image == 127)
        n_white = np.sum(segmented_image == 255)
        # Definir variáveis de interesse
        cloud_percentage = n_gray/(n_gray+n_white)*100
        cf_asi = int(round(cloud_percentage,0))
        return n_gray, n_white, n_black, cf_asi

    @staticmethod
    def plot_hist_RGB(img, mask, date_str, date_ts, color, path_out):
        '''
        Plot histogram from a RGB image
        '''
        # Separar cores
        blue_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        red_channel = img[:, :, 2]
        # Aplicar máscara para deixar preto os pontos que não interessam
        blue_channel = cv2.bitwise_and(blue_channel, blue_channel, mask=mask)
        green_channel = cv2.bitwise_and(green_channel, green_channel, mask=mask)
        red_channel = cv2.bitwise_and(red_channel, red_channel, mask=mask)
        # Remover pontos pretos, pois não fazem parte do céu/nuvem
        blue_channel_without_zero = blue_channel[blue_channel != 0]
        green_channel_without_zero = green_channel[green_channel != 0]
        red_channel_without_zero = red_channel[red_channel != 0]
        imgs_channels = [blue_channel_without_zero, 
                        green_channel_without_zero, 
                        red_channel_without_zero]
        # Escolher canal para calcular índices: G=1, R=2
        if color == 'green':
            img_channel = green_channel
            img_without_zero = green_channel_without_zero
        elif color == 'red':
            img_channel = red_channel
            img_without_zero = red_channel_without_zero

        # CÁLCULO DE ENTROPIA
        # Calcule o histograma
        histograma = cv2.calcHist([img_without_zero], [0], None, [256], [0, 256])
        # Normalize o histograma
        probabilidades = histograma / np.sum(histograma)
        # Calcule a entropia
        entropia = -np.sum(probabilidades * np.log2(probabilidades + np.finfo(float).eps))

        # CÁLCULO DE HTW
        # Encontre os pontos onde a área acumulada atinge 25% e 75% da área total
        cumsum = np.cumsum(probabilidades)
        x_25 = np.argwhere(cumsum >= 0.25)[0][0]
        x_75 = np.argwhere(cumsum >= 0.75)[0][0]
        # Calcule a HTW
        htw = abs(x_75 - x_25)

        # Fazer gráfico
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_channel, cmap='gray')
        ax[0].set_title(f'Imagem canal {color} com máscara - {date_ts} UTC')
        ax[0].set_axis_off()
        # Define colors to plot the histograms
        colors = ('b','g','r')
        for i,color in enumerate(colors):
            # Compute and plot histograms
            hist = cv2.calcHist([imgs_channels[i]],[0],None,[256],[0,256])
            ax[1].plot(hist,color = color)
        ax[1].set_title(f'Histograma - Entropia:{entropia:.2f} HTW:{htw}')
        # Crie um segundo eixo y à direita
        #ax2 = ax[1].twinx()
        # Plote a PDF no segundo eixo y
        #ax2.plot(x, pdf, label='PDF', color='r')
        # Configuração do segundo eixo y
        #ax2.set_ylim(0, max(pdf))  # Define o intervalo do eixo y à direita como 0 a 1
        #ax2.set_ylabel('PDF', color='r')  # Rótulo do eixo y à direita
        #ax2.tick_params(axis='y', labelcolor='r')  # Cor dos números do eixo y à direita
        fig.tight_layout()
        plt.savefig(f'{path_out}/{date_str}_hist.png', bbox_inches='tight')
        plt.close(fig)
        return entropia, htw

    @staticmethod
    def plot3(img1, img2, mask, date_str, n_gray, cf_asi, circ_info, path_out):
        '''
        Plotar três imagens lado a lado
        '''
        # Plotar círculo verde
        xc = circ_info[0]
        yc = circ_info[1]
        rc = circ_info[2]
        cv2.circle(img1, (xc, yc), rc, (0, 255, 0), 2)
        # Converter BGR para RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        # Cria uma figura com duas sub-figuras lado a lado
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # Plota a primeira imagem na primeira sub-figura
        ax1.imshow(img1)
        ax1.axis('off')  # Desliga os eixos
        ax1.set_title('Imagem RGB + Limite ângulo zenital')
        # Plota a segunda imagem na segunda sub-figura
        ax2.imshow(mask, cmap='gray')
        ax2.axis('off')  # Desliga os eixos
        ax2.set_title('Máscara')
        # Plota a terceira imagem na segunda sub-figura
        ax3.imshow(img2, cmap='gray')
        ax3.axis('off')  # Desliga os eixos
        ax3.set_title('Máscara + Imagem segmentada')

        # Adiciona o título geral à figura
        plt.suptitle(f'{n_gray} - {cf_asi}%', fontsize=16)
        # Ajusta o layout para garantir que nada seja cortado
        plt.tight_layout()
        plt.savefig(f'{path_out}/{date_str}_cf.png')
        plt.close(fig)
