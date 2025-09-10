#!/usr/bin/env python3.9.17
# -*- Coding: UTF-8 -*-

'''
Helper functions for handling files and other tasks
author: Vinicius Roggério da Rocha
e-mail: vinicius.rocha@inpe.br
version: 0.0.1
date: 2023-07-10
'''

import os
import tarfile
from glob import glob
from datetime import datetime, time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Aux():

    @staticmethod
    def pc_paths(pc_name):
        '''
        Define caminhos de arquivos conforme computador
        '''
        if pc_name == 'matrix':
            # Notebook Vinicius
            path_in = '/data1/tsiskyimage/'
            path_out = '/home/vinicius/Documentos/doutorado/out/'
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
    def csvs_to_df(path, endswith):
        '''
        Junta todos os arquivos CSVs de um diretório, que terminem com um padrão,
        em uma só dataframe
        '''
        dataframes = []
        files = sorted([f for f in glob(path + "/*" + endswith, recursive=True)])
        # Loop pelos arquivos no diretório
        for filename in files:
            # Lê o arquivo CSV e adiciona ao DataFrame à lista
            df = pd.read_csv(filename)
            dataframes.append(df)
        # Combina todos os DataFrames em um único DataFrame
        resultado = pd.concat(dataframes, ignore_index=True)
        return resultado
    
    @staticmethod
    def find_pattern(string_procurada, lista_original):
        '''
        Selecionar elementos de uma lista que tenham um dado padrão
        '''
        sublista = [item for item in lista_original if string_procurada in item]
        return sublista

    @staticmethod
    def dates_lst(lst):
        '''
        Extrair datas de strings em lista
        e gerar um vetor sem repetições
        '''
        lst_final = []
        for item in lst:
            new_item = item.split('.')[-3]
            lst_final.append(new_item)
        lst_final = np.unique(lst_final)
        return lst_final

    @staticmethod
    def check_time(date_dt, hi, mi, hf, mf):
        '''
        Verificar se horário dado está dentro de intervalo
        date_dt: em UTC
        '''
        start = time(hi, mi)
        end = time(hf, mf)
        if start <= date_dt.time() <= end:
            return 'high'
        else:
            return 'low'

    @classmethod
    def extract_tar(self, input_directory, output_directory):
        '''
        Extract TAR files content
        '''
        # Lista os arquivos .tar no diretório de entrada
        tar_files = self.list_files(input_directory, 'tar')

        for tar_file in tar_files:
            # Extrai o nome da pasta da substring do nome do arquivo .tar
            folder_name = tar_file.split('.')[2]
            # Cria o caminho completo para o diretório de saída
            output_path = os.path.join(output_directory, folder_name)
            # Cria o diretório de saída se ainda não existir
            os.makedirs(output_path, exist_ok=True)
            # Caminho completo para o arquivo .tar de entrada
            input_tar_path = os.path.join(input_directory, tar_file)
            # Extrai o conteúdo do arquivo .tar para o diretório de saída
            with tarfile.open(input_tar_path, 'r') as tar:
                tar.extractall(path=output_path)
            
        print(f'Extraido {len(tar_files)} arquivos TAR')
        exit()

    @staticmethod
    def rename_files(diretorio):
        '''
        Renomeia os arquivos extraídos - nunca testada
        '''
        for diretorio_raiz, subdiretorios, arquivos in os.walk(diretorio):
            for arquivo in arquivos:
                localtipo = arquivo.split('.')[0]
                datahora = arquivo.split('.')[-2]
                novo_nome = f'{localtipo}_{datahora}.jpg'
                novo_caminho = os.path.join(diretorio_raiz, novo_nome)
                antigo_caminho = os.path.join(diretorio_raiz, arquivo)
                #os.rename(antigo_caminho, novo_caminho)
                print(f"Renomeado: {antigo_caminho} -> {novo_caminho}")
                #exit()

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
    def datetime_str(filename):
        '''
        Get timestamp string and convert format
        '''
        date_str = filename.split('/')[-1].split('.')[-3]
        time_str = filename.split('/')[-1].split('.')[-2]
        date_dt = datetime.strptime(date_str+time_str, '%Y%m%d%H%M%S')
        date_ts = datetime.strftime(date_dt, '%Y-%m-%d %H:%M:%S')
        return date_str+time_str, date_dt, date_ts

    @staticmethod
    def get_date(filename):
        '''
        Get date string
        '''
        date_str = filename.split('/')[-1].split('.')[-2].split('_')[1]
        return date_str

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
    
    @staticmethod
    def create_df(col_names):
        '''
        Create dataframe with given column names
        '''
        df = pd.DataFrame(columns=col_names)
        return df
    
    @staticmethod
    def create_df_cont(df_data, col_names):
        '''
        Create dataframe with given content
        '''
        df = pd.DataFrame(df_data, columns=col_names)
        return df
    
    @staticmethod
    def concat_df(df1, df2):
        '''
        Concat two pandas dataframes
        '''
        df = pd.concat([df1, df2], ignore_index=True)
        return df

    @staticmethod
    def merge_df(df1, df2):
        '''
        Merge two pandas dataframes by Timestamp column
        '''
        # Convertendo os valores da coluna 'timestamp' para datetime
        df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
        df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
        # Juntando os DataFrames usando a coluna 'timestamp' como chave
        merged_df = pd.merge(df1, df2, on='Timestamp')
        return merged_df

    @staticmethod
    def remove_rows(df, col_name, val):
        '''
        Delete rows based on column value
        '''
        df.drop(df[df[col_name] == val].index, inplace = True)
        return df

    @staticmethod
    def df_to_csv(df, filename):
        '''
        Save a pandas dataframes to CSV file
        '''
        df.to_csv(filename, index=False)

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
    def load_csv(filename):
        '''
        Load CSV to pandas dataframe
        '''
        df = pd.read_csv(filename)
        return df

    @staticmethod
    def sub_df(df, col_name, condition):
        '''
        Select rows based on condition
        Convert string to datetime
        '''
        df_sel = df.loc[df[col_name] == condition]
        # Zerar índices e gerar cópia de df
        df_sel = df_sel.reset_index(drop=True).copy()
        # Convertendo a coluna 'Timestamp' para objetos datetime
        df_sel['Timestamp'] = pd.to_datetime(df_sel['Timestamp'].astype(str), format='%Y%m%d%H%M%S')
        return df_sel

    @staticmethod
    def open_nc(filename):
        '''
        Open nc file
        '''
        ds = xr.open_dataset(filename)
        return ds
    
    @staticmethod
    def view_nc(ds):
        '''
        Print variables - 1st field and long_name are importants
        '''
        # Check if has lat/lon or not and print
        if hasattr(ds, 'lat'):
            print(f"Lat/Lon = {ds.variables['lat'].values},{ds.variables['lon'].values}")
        # Print variables
        for item in ds.variables.items():
            print(item)
            print('-------------------------------------------------')
        exit()

    @staticmethod
    def concat_ds(lista_de_arquivos):
        '''
        Concatenar datasets dos arquivos da lista
        '''
        # Inicializar uma lista vazia para armazenar os datasets individuais
        datasets = []
        # Loop para abrir cada arquivo NetCDF e adicionar o dataset à lista
        for arquivo in lista_de_arquivos:
            dataset = xr.open_dataset(arquivo)
            datasets.append(dataset)
        # Concatenar os datasets ao longo da dimensão desejada (por exemplo, tempo)
        dataset_concatenado = xr.concat(datasets, dim='time')
        # Fechar os datasets individuais
        for dataset in datasets:
            dataset.close()
        return dataset_concatenado

    @staticmethod
    def extract_ts(ds):
        '''
        Extrair séries temporais de um dataset
        '''
        xtime = ds.variables['time'][:].values
        percent_opaque = ds.variables['percent_opaque'][:].values
        percent_thin = ds.variables['percent_thin'][:].values
        cf = percent_opaque + percent_thin
        return xtime, cf

    @staticmethod
    def more_data(dates_str):
        '''
        Find the file with more date
        '''
        # Inicializa as variáveis para armazenar as informações do arquivo com maior tamanho
        maior_tamanho = 0
        nome_maior_arquivo = ''
        quantidade_de_dados = 0
        # Itera sobre as strings da lista
        for date_str in dates_str:
            # Forma o caminho completo para o arquivo CDF
            caminho_arquivo = f'/data1/sonde/maosondewnpnM1.b1.{date_str[:8]}.{date_str[8:]}.cdf'
            # Carrega o arquivo CDF como um dataset xarray
            ds = xr.open_dataset(caminho_arquivo)
            
            # Obtém o tamanho da variável "pressure"
            tamanho_pressure = ds['pres'].size
            
            # Verifica se o tamanho atual é maior do que o maior tamanho encontrado até agora
            if tamanho_pressure > maior_tamanho:
                maior_tamanho = tamanho_pressure
                nome_maior_arquivo = caminho_arquivo
            
            # Fecha o dataset
            ds.close()
        return nome_maior_arquivo

    @staticmethod
    def path_names(lst_str):
        '''
        Merge strings to create file paths
        '''
        filenames = []
        for date_str in lst_str:
            path = f'/data1/sonde/maosondewnpnM1.b1.{date_str[:8]}.{date_str[8:]}.cdf'
            filenames.append(path)
        return filenames
    
    @staticmethod
    def plot_daily(df, date_str):
        '''
        Plot daily cicle from variable
        '''
        # Converter a coluna Timestamp para o formato de data e hora
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')

        # Definir a ordem desejada para os elementos no eixo Y
        ordem_class_sky = ['1_cumulus', '2_altocumulus', '3_cirrus', '4_clearsky', '5_stratocumulus', '6_cumulonimbus', '7_mixed', 'mask_not_ok']
        # Criar a categoria ordenada para a coluna Class_Sky
        df['Class_Sky'] = pd.Categorical(df['Class_Sky'], categories=ordem_class_sky, ordered=True)
        # Ordenar o DataFrame com base na nova coluna Class_Sky
        df = df.sort_values('Class_Sky')

        # Criar o gráfico
        plt.figure(figsize=(10, 6))
        # Plotar os dados
        plt.plot(df['Timestamp'], df['Class_Sky'], marker='o', linestyle='', markersize=5)
        # Configurar rótulos e título do gráfico
        plt.ylabel('Classe')
        plt.title(f'Classes - {date_str}')
        # Configurar os rótulos do eixo x no formato HH:MM
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Rotacionar rótulos do eixo x para melhor visualização
        plt.xticks(rotation=45)

        # Salvar o gráfico em um arquivo PNG
        plot_name = f'output_testes/{date_str}.png'
        plt.savefig(plot_name, bbox_inches='tight')

    @staticmethod
    def plot_dif_day(df, date_str, time_column, column1, column2, plot_name):
        '''
        Gráfico de diferenças entre colunas
        '''
        # Convert string values to NaN in the specified columns
        df[column1] = pd.to_numeric(df[column1], errors='coerce')
        df[column2] = pd.to_numeric(df[column2], errors='coerce')
        # Convert negative values (-100 = sem algoritmo TSI)
        df[column1] = df[column1].apply(lambda x: np.nan if x < 0 else x)
        df[column2] = df[column2].apply(lambda x: np.nan if x < 0 else x)
        # Extract data from the DataFrame
        time = df[time_column]
        diff = df[column1] - df[column2]
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(time, diff, marker='o', linestyle='', color='b')
        # Format x-axis labels to show HH:MM
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xlabel(time_column)
        plt.ylabel(f"{column1} - {column2}")
        plt.title(f"Difference between {column1} and {column2} - {date_str}")
        plt.grid(True)
        plt.xticks(rotation=45)
        # Salvar o gráfico em um arquivo PNG
        plt.savefig(plot_name, bbox_inches='tight')

    @staticmethod
    def plot_msd(df, plot_name, n_plots):
        '''
        Plot a time series with Mean and Standard Deviation;
        If n_plots == 2, add one more time series to the graph
        '''
        # Converter a coluna Timestamp para o formato de data e hora
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d')
        # Montar gráfico
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Timestamp'], df['CF_CALC_m'], yerr=df['CF_CALC_sd'],
                    ecolor='gray', fmt='o', markersize=4, label='CF_CALC')
        if n_plots == 2:
            plt.errorbar(df['Timestamp'], df['CF_TSI_m'], yerr=df['CF_TSI_sd'],
                        ecolor='gray', fmt='o', markersize=4, label='CF_TSI')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Fração de cobertura de nuvem - médias diárias')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_name, bbox_inches='tight')

    @staticmethod
    def plot_comp(df, plot_name):
        '''
        Plot varX x varY comparison
        '''
        plt.figure(figsize=(8, 6))
        plt.scatter(df['CF_TSI'], df['Cloud_Fraction'], label='Dados')
        plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='y = x')
        plt.xlabel('CF_TSI')
        plt.ylabel('Cloud_Fraction')
        plt.title('Gráfico de CF_TSI vs Cloud_Fraction')
        plt.legend()
        plt.grid(True)
        # Salvar o gráfico em um arquivo PNG
        plt.savefig(plot_name, bbox_inches='tight')

    @staticmethod
    def select_mont(lista, year, month):
        '''
        Select filenames from same year-month
        '''
        nova_lista = []
        for string in lista:
            substring = string.split('/')[-1].split('.')[-2].split('_')[1]
            if substring[:4] == year and substring[4:6] == month:
                nova_lista.append(string)
        return nova_lista

    @staticmethod
    def stats_month(csv_file_list):
        '''
        Estatísticas mensais
        '''
        # Inicializa um dataframe vazio
        combined_df = pd.DataFrame()
        # Loop através da lista de arquivos CSV
        for csv_file in csv_file_list:
            # Lê o arquivo CSV em um dataframe temporário
            temp_df = pd.read_csv(csv_file)
            # Concatena o dataframe temporário ao dataframe combinado
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        # Calcula o número de ocorrências de cada classe
        class_counts = combined_df['Class_Sky'].value_counts()
        # Calcula os percentuais de ocorrência de cada classe
        class_percentages = (class_counts / class_counts.sum()) * 100
        # Cria um novo dataframe com as contagens e os percentuais
        class_occurrences = pd.DataFrame({'Count': class_counts, 'Percentage': class_percentages})
        # Calcula o total de ocorrências e percentuais
        total_count = class_counts.sum()
        total_percentage = 100.0
        # Adiciona uma linha com os totais ao dataframe
        total_row = pd.Series({'Count': total_count, 'Percentage': total_percentage}, name='Total')
        class_occurrences = class_occurrences.append(total_row)
        return class_occurrences
    
    @staticmethod
    def calc_stats(df, day_str, column1, column2):
        '''
        Calcula estatísticas para comparar valores
        desconsiderando valores negativos
        '''
        #differences = df[column1] - df[column2]
        #mean_difference = differences.mean()
        #std_difference = differences.std()
        c1_nonzero = df[df[column1] >= 0][column1]
        c2_nonzero = df[df[column2] >= 0][column2]
        TSI_m = c1_nonzero.mean()
        TSI_sd = c1_nonzero.std()
        CF_m = c2_nonzero.mean()
        CF_sd = c2_nonzero.std()
        neg_CALC = (df[column1] < 0).sum()
        neg_TSI = (df[column2] < 0).sum()
        n_total = len(df[column1])
        return [day_str, CF_m, CF_sd, TSI_m, TSI_sd, neg_CALC, neg_TSI, n_total]

    @staticmethod
    def plot_ts(df, varname, date_str, plot_name):
        '''
        Plota o gráfico de uma variável em função do tempo
        '''
        # Converter a coluna Timestamp para o formato de data e hora
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Montar gráfico
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Timestamp'], df[varname], 
                      fmt='o', markersize=4, label='')
        #plt.xlabel('Timestamp')
        # Format x-axis labels to show HH:MM
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.ylabel(varname)
        plt.title(f'{varname} - {date_str}')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Salvar o gráfico em um arquivo PNG
        plt.savefig(plot_name, bbox_inches='tight')

    @staticmethod
    def plot_ts2(df, varname1, varname2, date_str, plot_name):
        '''
        Plota o gráfico de duas variáveis em função do tempo
        '''
        # Converter a coluna Timestamp para o formato de data e hora
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Cria uma figura com dois eixos y
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Eixo y esquerda
        ax1.plot(df['Timestamp'], df[varname1], label=varname1, color='tab:blue',
                 linestyle='', marker='.')
        ax1.set_ylabel(varname1, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        # Eixo y direita
        ax2 = ax1.twinx()
        ax2.plot(df['Timestamp'], df[varname2], label=varname2, color='tab:red',
                 linestyle='', marker='.')
        ax2.set_ylabel(varname2, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        # Formata os horários no eixo y no formato HH:MM
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Finaliza gráfico
        plt.title(f'{varname1} e {varname2} - {date_str}')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Salvar o gráfico em um arquivo PNG
        plt.savefig(plot_name, bbox_inches='tight')