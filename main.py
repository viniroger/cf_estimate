# Script para estimar valor de CF para todas as imagens de um diretório

from helpers.cf_estimate import Aux
from helpers.cf_estimate import Img_func

filetype = 'sonda' # opções: tsi, sonda
pc_name = 'matrix' # opções: matrix, rocinante, desktop-pedro
path_in, path_out = Aux.pc_paths(pc_name)

# Loop para todos os diretórios - cada um é um dia de dados
lst_dir = Aux.list_directories(path_in)
for dir_date in lst_dir:
    # Listar ASIs do dia
    #dir_date = '20141003' # testes
    print(f'*********{dir_date}**********')
    dir_in = path_in + dir_date
    files = Aux.list_files(f'{dir_in}/', 'jpg')
    # Criar tabela com dados
    df_clouds = Aux.create_df(['Timestamp', 'Cloud_Fraction', 'Class_Sky'])
    # Criar tabela com estatísticas
    df_stats = Aux.create_df(['Timestamp', 'Mask1_0', 'Mask1_255', 'Entropia', 'HTW',
                             'Homogeneidade', 'Iluminacao', 'N_cloud', 'N_clearsky', 'N_Other'])
    # Loop para todos os arquivos do diretório - 1 dia de dados
    for filename in files:
        #filename = '/data1/tsiskyimage/20141003/maotsiskyimageM1.a1.20141003.110000.jpg.20141003110000.jpg'
        #filename = '/data1/tsiskyimage/20141003/maotsiskyimageM1.a1.20141003.212800.jpg.20141003212800.jpg'
        #filename = '/home/vinicius/Documentos/doutorado/cf_tests_input/20200731201000_11.jpg'
        # Obter informações de data/horário através do nome do arquivo
        date_str, date_dt, date_ts = Aux.date_str(filename, filetype)
        print(date_ts)
        # Verificar horário (canal R/G para baixa iluminação) <100W/m2
        #illumination = Aux.check_time(date_dt, 10, 30, 20, 0)
        # Ler imagem
        img = Img_func.read_img(filename)
        # Criar máscara circular
        mask, counts_mask, circ_info = Img_func.create_mask(img, filetype)
        # Se a máscara retornou como string, imprimir erro e continuar loop
        res = isinstance(mask, str)
        if res:
            print(mask)
            clouds_info = [date_ts, -99, res]
            stat_info = [date_ts, counts_mask.get(0), counts_mask.get(255), 
                         -99, -99, 'None', 'None', -99, -99, -99]
            continue
        else:
            # Verificar qualidade da imagem através da máscara
            if filetype == 'tsi':
                flag = Aux.check_mask(counts_mask, 59000, 80000)
            else:
                flag = 'mask_ok'
            if flag == 'mask_ok':
                if filetype == 'tsi':
                    # Recortar imagem e máscara para virar quadrado
                    img, mask, circ_info = Img_func.crop_circle_center(img, mask, circ_info)
                # Verificar quantidade de amarelo
                #p_yellow = Img_func.find_yellow_hsv(img, mask)
                p_yellow = Img_func.find_yellow_lab(img, mask)
                # Critério de iluminação - ideia de Rashid et al (2021)
                limit_illumination = 10
                if p_yellow > limit_illumination:
                    # Ajustar temperatura da imagem (em vez de usar 1 só canal)
                    #img_bkp = img
                    img = Img_func.change_temperature(img, path_out, date_str)
                    #Img_func.plot_yellow(img_bkp, img, mask, p_yellow, date_str, path_out)
                # Calcular NBRR (1 canal só fica muito ruim)
                img_grayscale, nbrr = Img_func.NBRR(img)
                entropia, htw = Img_func.calc_entropia_htw(img_grayscale, mask)
                #Img_func.plot_hist_RGB(img, mask, date_str, date_ts, 'red', path_out)
                # Verificar se imagem é homogênea ou heterogênea
                # e obter parâmetros de MCE adaptativo
                criterio_homogeneidade, block_size, constant = Img_func.homog(entropia, htw)
                # Segmentar e calcular CF - Li et al. (2011) citado por Rashid et al (2021)
                seg_img = Img_func.cld_seg(img_grayscale, mask, block_size, constant)
                n_gray, n_white, n_black, cf_asi = Img_func.calc_cf(seg_img)
                # Classificar imagem
                #class_img = Ml_func.classify_image(masked_image)
                class_img = 'development'
                # Finalizar lista/linha de informações da ASI
                clouds_info = [date_ts, cf_asi, class_img]
                stat_info = [date_ts, counts_mask.get(0), counts_mask.get(255), entropia, htw,
                            criterio_homogeneidade, p_yellow, n_gray, n_white, n_black]
                # Plotar imagem para comparação
                #Img_func.plot3(img, seg_img, mask, date_str, n_gray, cf_asi, circ_info, path_out)
            else:
                clouds_info = [date_ts, -99, flag]
                stat_info = [date_ts, counts_mask.get(0), counts_mask.get(255),
                             -99, -99, 'None', -99, -99, -99, -99]
        # Atualizar df
        df_clouds = Aux.row_to_df(df_clouds, clouds_info)
        df_stats = Aux.row_to_df(df_stats, stat_info)
        #exit()

    # Salvar tabela com as datas e classificações de cada ASI
    Aux.save_df(df_clouds, f'{path_out}/ASIs_info/{dir_date}_clouds.csv')
    Aux.save_df(df_stats, f'{path_out}/ASIs_info/{dir_date}_stats.csv')
    #exit()
