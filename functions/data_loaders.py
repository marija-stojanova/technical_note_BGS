import pandas as pd

def load_tictoc_dataset(istep_only:bool=False, 
                        vt_only:bool=False, 
                        re7_only:bool=False, 
                        only_carbonated:bool=False, 
                        only_noncarbonated:bool=False,
                        TIC_carbonated_threshold:float=2):
    '''
    Load the complete dataset needed for TIC-TOC analyses:
    -LAS is the Laboratoire d'Analyses de Sol data (CHN analysis)
    -VT is Vinci Technologies data (RE6 and RE7 analysis)
    -Merieux is Merieux Laboratoires data (CHN analysis)
    -ISTeP is Institut des Sciences de la Terre de Paris data (RE6 analysis)

    if only_carbonated (only noncarbonated) then only the samples with a TIC > TIC_carbonated_threshold (TIC <= TIC_carbonated_threshold) are returned
    '''

    ### LAS CHN data
    las1 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00001_C.xlsx', engine='openpyxl')
    las2 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00002_C.xlsx', engine='openpyxl')
    las3 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00003_C.xlsx', engine='openpyxl')
    las4 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00004_C.xlsx', engine='openpyxl')
    las5 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00005_C.xlsx', engine='openpyxl')
    las6 = pd.read_excel('data/LAS ARRAS/R-LAS-2023-06-00006_C.xlsx', engine='openpyxl')

    las = pd.concat([las1, las2, las3, las4, las5, las6])
    las = las.loc[:,~las.columns.str.match("Unnamed")]
    las['TIC_las'] = las['C'] - las['TOC']
    las.rename(columns={'TOC' : 'TOC_las', 'N' : 'N_las'}, inplace=True)

    ### ISTeP RE6 data
    istep = pd.read_csv('data/ISTeP/tic-toc-istep.csv', encoding='utf-8', sep=';')
    istep[['Prefix', 'Sample']] = istep['Analyse'].str.split('_', n=1, expand=True)
    istep['TOC_istep'] = istep['TOCre6']
    istep['TIC_istep'] = istep['MINC']*10

    ### VT RE6 data from raw files
    vt_raw = pd.read_csv('data/VT/tic-toc-vt.csv', sep=';')
    vt_raw[['Prefix', 'Sample']] = vt_raw['Analyse'].str.split('_', n=1, expand=True)
    vt_raw['TOC_vt'] = vt_raw['TOCre6']
    vt_raw['TIC_vt'] = vt_raw['MINC']*10
    vt_raw2 = pd.read_csv('data/VT/tic-toc-vt-redone_samples.csv', sep=';')
    vt_raw2[['Prefix', 'Sample']] = vt_raw2['Analyse'].str.split('_', n=1, expand=True)
    vt_raw2['TOC_vt'] = vt_raw2['TOCre6']
    vt_raw2['TIC_vt'] = vt_raw2['MINC']*10
    vt_raw.set_index('Sample', inplace=True)
    vt_raw2.set_index('Sample', inplace=True)
    vt_raw.update(vt_raw2)

    nitrogen_genesis = pd.read_csv('data/TICTOC GENESIS DATA/azote_genesis.csv')
    nitrogen_genesis['sample_id'] = nitrogen_genesis['sample_id'].astype(str)
    nitrogen_genesis.rename(columns={'TN_CHN' : 'N_merieux'}, inplace=True)

    ### VT RE7 data
    vt_re7 = pd.read_csv('data/VT/tic-toc-vt-re7.csv', sep=';')
    vt_re7[['Prefix', 'Sample']] = vt_re7['Analyse'].str.split('_', n=1, expand=True)
    vt_re7['TOC_re7'] = vt_re7['TOCre6']
    vt_re7['TIC_re7'] = vt_re7['MINC']*10
    vt_re7 = vt_re7.loc[:,~vt_re7.columns.str.match("Unnamed")]


    ### Merieux CHN data
    genesis = pd.read_excel('data/TICTOC GENESIS DATA/2023-04-14_UPDATED_selected_Genesis_samples_complementary_analyses.xlsx', engine='openpyxl')
    merieux = pd.DataFrame()
    merieux['TOC_merieux'] = genesis['TOC_CHN']
    merieux['TIC_merieux'] = genesis['TCaCO3'] * 0.120002597739966
    merieux['Sample'] = genesis['id']

    # Clean up and merge everything together
    las['Sample'] = las['Sample'].astype(str)
    merieux['Sample'] = merieux['Sample'].astype(str)
    nitrogen_genesis['sample_id'] = nitrogen_genesis['sample_id'].astype(str)
    istep['Sample'] = istep['Sample'].astype(str)
    vt_re7['Sample'] = vt_re7['Sample'].astype(str)
    
    merieux = pd.merge(left=merieux, left_on='Sample', right=nitrogen_genesis, right_on='sample_id')
    all = pd.merge(left = merieux[['Sample', 'TIC_merieux', 'TOC_merieux', 'N_merieux']], left_on='Sample', right=las[['Sample', 'TIC_las', 'TOC_las', 'N_las']], right_on='Sample')
    
    if istep_only:
        all = pd.merge(left=all, left_on='Sample', right=istep, right_on='Sample')
    elif vt_only:
        all = pd.merge(left=all, left_on='Sample', right=vt_raw, right_on='Sample')
    elif re7_only:
        all = pd.merge(left=all, left_on='Sample', right=vt_re7, right_on='Sample')
    else:
        all = pd.merge(left=all, left_on='Sample', right=istep, right_on='Sample')
        all = pd.merge(left=all, left_on='Sample', right=vt_raw, right_on='Sample', suffixes=['_istep', '_vt'])
        all = pd.merge(left=all, left_on='Sample', right=vt_re7[['Sample', 'TOC_re7', 'TIC_re7']], right_on='Sample')

    all=all.dropna()

    # Add the columns needed for modeling
    all['CoverN_las'] = (all['TOC_las'] + all['TIC_las'])/all['N_las']
    all['CoverN_merieux'] = (all['TOC_merieux'] + all['TIC_merieux'])/all['N_merieux']
    all['C_merieux'] = all['TIC_merieux'] + all['TOC_merieux']
    all['C_las'] = all['TIC_las'] + all['TOC_las']

    all['delta_toc_las_merieux'] = all['TOC_las'] - all['TOC_merieux']
    all['delta_tic_las_merieux'] = all['TIC_las'] - all['TIC_merieux']

    all['delta_c_las_merieux'] = all['C_las'] - all['C_merieux']

    if istep_only:
        # There are two inverted ISTeP values, correct them
        all_copy = all.copy()
        all.loc[all['Sample'] == '5172', 'TOC_istep'] = all_copy.loc[all_copy['Sample'] == '5162']['TOC_istep'].item()
        all.loc[all['Sample'] == '5172', 'TIC_istep'] = all_copy.loc[all_copy['Sample'] == '5162']['TIC_istep'].item()
        all.loc[all['Sample'] == '5162', 'TOC_istep'] = all_copy.loc[all_copy['Sample'] == '5172']['TOC_istep'].item()
        all.loc[all['Sample'] == '5162', 'TIC_istep'] = all_copy.loc[all_copy['Sample'] == '5172']['TIC_istep'].item()

        all['C_istep'] = all['TIC_istep'] + all['TOC_istep']
        all['delta_toc_las_istep'] = all['TOC_las'] - all['TOC_istep']
        all['delta_tic_las_istep'] = all['TIC_las'] - all['TIC_istep']
        all['delta_toc_merieux_istep'] = all['TOC_merieux'] - all['TOC_istep']
        all['delta_tic_merieux_istep'] = all['TIC_merieux'] - all['TIC_istep']

        if only_carbonated:
            all = all[all['TIC_istep'] > TIC_carbonated_threshold]
        elif only_noncarbonated:
            all = all[all['TIC_istep'] <=  TIC_carbonated_threshold]

    elif vt_only:
        all['C_vt'] = all['TIC_vt'] + all['TOC_vt']
        all['delta_toc_las_vt'] = all['TOC_las'] - all['TOC_vt']
        all['delta_tic_las_vt'] = all['TIC_las'] - all['TIC_vt']
        all['delta_toc_merieux_vt'] = all['TOC_merieux'] - all['TOC_vt']
        all['delta_tic_merieux_vt'] = all['TIC_merieux'] - all['TIC_vt']
        if only_carbonated:
            all = all[all['TIC_vt'] > TIC_carbonated_threshold]
        elif only_noncarbonated:
            all = all[all['TIC_vt'] <=  TIC_carbonated_threshold]

    elif re7_only:
        all['C_re7'] = all['TIC_re7'] + all['TOC_re7']
        all['delta_toc_las_re7'] = all['TOC_las'] - all['TOC_re7']
        all['delta_tic_las_re7'] = all['TIC_las'] - all['TIC_re7']
        all['delta_toc_merieux_re7'] = all['TOC_merieux'] - all['TOC_re7']
        all['delta_tic_merieux_re7'] = all['TIC_merieux'] - all['TIC_re7']
        if only_carbonated:
            all = all[all['TIC_re7'] > TIC_carbonated_threshold]
        elif only_noncarbonated:
            all = all[all['TIC_re7'] <=  TIC_carbonated_threshold]

    else:
        # There are two inverted ISTeP values, correct them
        all_copy = all.copy()
        all.loc[all['Sample'] == '5172', 'TOC_istep'] = all_copy.loc[all_copy['Sample'] == '5162']['TOC_istep'].item()
        all.loc[all['Sample'] == '5172', 'TIC_istep'] = all_copy.loc[all_copy['Sample'] == '5162']['TIC_istep'].item()
        all.loc[all['Sample'] == '5162', 'TOC_istep'] = all_copy.loc[all_copy['Sample'] == '5172']['TOC_istep'].item()
        all.loc[all['Sample'] == '5162', 'TIC_istep'] = all_copy.loc[all_copy['Sample'] == '5172']['TIC_istep'].item()

        all['delta_toc_istep_vt'] = all['TOC_istep'] - all['TOC_vt']
        all['delta_tic_istep_vt'] = all['TIC_istep'] - all['TIC_vt']
        all['C_istep'] = all['TIC_istep'] + all['TOC_istep']
        all['delta_toc_las_istep'] = all['TOC_las'] - all['TOC_istep']
        all['delta_tic_las_istep'] = all['TIC_las'] - all['TIC_istep']
        all['delta_toc_merieux_istep'] = all['TOC_merieux'] - all['TOC_istep']
        all['delta_tic_merieux_istep'] = all['TIC_merieux'] - all['TIC_istep']
        all['C_vt'] = all['TIC_vt'] + all['TOC_vt']
        all['delta_toc_las_vt'] = all['TOC_las'] - all['TOC_vt']
        all['delta_tic_las_vt'] = all['TIC_las'] - all['TIC_vt']
        all['delta_toc_merieux_vt'] = all['TOC_merieux'] - all['TOC_vt']
        all['delta_tic_merieux_vt'] = all['TIC_merieux'] - all['TIC_vt']
        all['C_re7'] = all['TIC_re7'] + all['TOC_re7']
        all['delta_toc_las_re7'] = all['TOC_las'] - all['TOC_re7']
        all['delta_tic_las_re7'] = all['TIC_las'] - all['TIC_re7']
        all['delta_toc_merieux_re7'] = all['TOC_merieux'] - all['TOC_re7']
        all['delta_tic_merieux_re7'] = all['TIC_merieux'] - all['TIC_re7']
        

        if only_carbonated:
            all = all[all['TIC_istep'] > TIC_carbonated_threshold]
        elif only_noncarbonated:
            all = all[all['TIC_istep'] <=  TIC_carbonated_threshold]

    all.drop(all[all.Sample.isin(['26301', '26302', '26303', '9409', '9134', '23700', '11604', '24617', '24618', '24619'])].index, 
             axis=0, 
             errors='ignore', 
             inplace=True)

    return all


def load_rmqs(only_carbonated:bool=False, 
              only_noncarbonated:bool=False,
              yield_check:bool=False,
              TIC_carbonated_threshold:float=2,
              remove_TIC_outliers:bool=True,
              TIC_outlier_threshold:float=5):
    rmqs = pd.read_csv('data/RMQS/FulldataG.csv', sep=';', encoding='unicode_escape', decimal=',')

    rmqs.rename(columns={
                'Sample' : 'ID',
                'argile' : 'clay',
                'InoC' : 'TIC_las',
                'C_total' : 'Total_c_las',
                'carbone_16_5_1' : 'TOC_las',
                'sable_total' : 'sand',
                'limon_total' : 'silt',
                'TOCRE6' : 'TOC_istep',
                'MINC' : 'TIC_istep',
                'OIRE6' : 'OIre6',
                'PC_TOCRE6' : 'PC_TOCre6',
                'S2_soil' : 'S2',
    }, inplace=True)

    rmqs['delta_toc_las_istep'] = rmqs['TOC_las'] - rmqs['TOC_istep']
    rmqs['delta_tic_las_istep'] = rmqs['TIC_las'] - rmqs['TIC_istep']

    rmqs_profondeur = pd.read_csv('data/RMQS_profondeur/rmqs-profondeur-re6.csv', sep=';')
    rmqs_profondeur_1 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_1.xlsx', engine='openpyxl')
    rmqs_profondeur_2 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_2.xlsx', engine='openpyxl')
    rmqs_profondeur_3 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_3.xlsx', engine='openpyxl')
    rmqs_profondeur_4 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_4.xlsx', engine='openpyxl')
    rmqs_profondeur_5 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_5.xlsx', engine='openpyxl')
    rmqs_profondeur_6 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_6.xlsx', engine='openpyxl')
    rmqs_profondeur_7 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_7.xlsx', engine='openpyxl')
    rmqs_profondeur_8 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_8.xlsx', engine='openpyxl')
    rmqs_profondeur_9 = pd.read_excel('data/RMQS_profondeur/RMQS_Prof_9.xlsx', engine='openpyxl')

    rmqs_profondeur_chn = pd.concat([rmqs_profondeur_1, rmqs_profondeur_2, rmqs_profondeur_3, rmqs_profondeur_4, rmqs_profondeur_5, rmqs_profondeur_6, rmqs_profondeur_7, rmqs_profondeur_8, rmqs_profondeur_9])
    rmqs_profondeur_chn['Analysis'] = rmqs_profondeur_chn['Analysis'].str.replace('.R00', '')
    rmqs_profondeur['Analyse'] = rmqs_profondeur['Analyse'].str.upper()
    rmqs_profondeur_chn['Analysis'] = rmqs_profondeur_chn['Analysis'].str.upper()
    
    rmqs_profondeur = pd.merge(left=rmqs_profondeur, left_on='Analyse', right=rmqs_profondeur_chn[['Corg(%)', 'CaCO3', 'Analysis']], right_on='Analysis')
    rmqs_profondeur['TIC_las'] = rmqs_profondeur['CaCO3'] * 0.120002597739966
    rmqs_profondeur['TOC_las'] = rmqs_profondeur['Corg(%)']
    rmqs_profondeur['TOC_istep'] = rmqs_profondeur['TOCre6']
    rmqs_profondeur['TIC_istep'] = rmqs_profondeur['MINC']*10
    rmqs_profondeur = rmqs_profondeur.dropna(subset=['TIC_las'])
    rmqs_profondeur = rmqs_profondeur.dropna(subset=['TOC_las'])
    rmqs_profondeur['delta_toc_las_istep'] = rmqs_profondeur['TOC_las'] - rmqs_profondeur['TOC_istep']
    rmqs_profondeur['delta_tic_las_istep'] = rmqs_profondeur['TIC_las'] - rmqs_profondeur['TIC_istep']

    if only_carbonated:
        rmqs_profondeur = rmqs_profondeur[rmqs_profondeur['TIC_istep'] > TIC_carbonated_threshold]
        rmqs = rmqs[rmqs['TIC_istep'] > 5]
    elif only_noncarbonated:
        rmqs_profondeur = rmqs_profondeur[rmqs_profondeur['TIC_istep'] <= TIC_carbonated_threshold]
        rmqs = rmqs[rmqs['TIC_istep'] <= 5]

    if yield_check:
        rmqs_profondeur['yield_istep_las'] = (rmqs_profondeur['TIC_istep'] + rmqs_profondeur['TOC_istep']) / (rmqs_profondeur['TIC_las'] + rmqs_profondeur['TOC_las'])
        rmqs_profondeur = rmqs_profondeur[(rmqs_profondeur['yield_istep_las'] >= 0.7) & (rmqs_profondeur['yield_istep_las'] <= 1.3)]
        rmqs_profondeur = rmqs_profondeur[rmqs_profondeur['TOC_las'] <= 60]

    if remove_TIC_outliers:
        rmqs_profondeur = rmqs_profondeur[(rmqs_profondeur['delta_tic_las_istep'] > -TIC_outlier_threshold) &
                                          (rmqs_profondeur['delta_tic_las_istep'] < TIC_outlier_threshold)]


    return rmqs, rmqs_profondeur