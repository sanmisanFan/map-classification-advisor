# -*- coding: utf-8 -*-
__author__ = 'Fan LEI'

import json
import geopandas
import mapclassify
import numpy as np
from libpysal.weights import Queen
from esda.moran import Moran

#Calculate GVF for the spatial data and given classification
def GVF(list, breaks, num_classes):
    mean_SDAM = sum(list)/len(list)
    Xdif2_SDAM = []
    for item in list:
        Xdif2_SDAM.append((item - mean_SDAM) ** 2)
    SDAM = sum(Xdif2_SDAM)

    class_mean_list = []
    temp_list = []
    for item in list:
        if item <= breaks[0]:
            temp_list.append(item)
    class_mean_list.append(sum(temp_list)/len(temp_list))
    del temp_list[:]

    for y in range(0, num_classes - 2):
        for item in list:
            if item > breaks[y] and item <= breaks[y+1]:
                temp_list.append(item)
        class_mean_list.append(sum(temp_list)/len(temp_list))
        del temp_list[:]
        y += 1

    for item in list:
        if item > breaks[num_classes - 2]:
            temp_list.append(item)
    class_mean_list.append(sum(temp_list)/len(temp_list))
    del temp_list[:]

    #calculate SDCM
    SDCM_list = []
    for item in list:
        if item <= breaks[0]:
            x_diff_2 = (item - class_mean_list[0]) ** 2
            SDCM_list.append(x_diff_2)

    for y in range(0, num_classes - 2):
        for item in list:
            if item > breaks[y] and item <= breaks[y+1]:
                x_diff_2 = (item - class_mean_list[y+1]) ** 2
                SDCM_list.append(x_diff_2)
        y += 1

    for item in list:
        if item > breaks[num_classes - 2]:
            x_diff_2 = (item - class_mean_list[num_classes - 1]) ** 2
            SDCM_list.append(x_diff_2)

    SDCM = sum(SDCM_list)

    #This is where the GVF value is calculated and displayed
    GVF = (SDAM - SDCM) / SDAM
    return GVF

def goodness_of_variance_fit(array, breaks):
    # get the break points
    classes = breaks

    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])

    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def isNaN(num):
    return num == num

def getEqualInterval(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        EI = mapclassify.EqualInterval(list, k=i)
        EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
        dict['breaks'] = EI_bins
        dict['adcm'] = EI.adcm
        dict['gadf'] = EI.gadf
        dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
        classList = EI.yb.tolist()
        makeGeoJson(classList, "EI", path, inputfeature, mx_json)
        moran_result = moranCal("EI")
        #print(moran_result)
        if(moran_result["moran_p"] > 0.05):
            dict['moran'] = -1
        else:
            dict['moran'] = moran_result['moran_I']
        EI_list.append(dict)
    return EI_list

def getQuantiles(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        EI = mapclassify.Quantiles(list, k=i)
        EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
        dict['breaks'] = EI_bins
        dict['adcm'] = EI.adcm
        dict['gadf'] = EI.gadf
        dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
        classList = EI.yb.tolist()
        makeGeoJson(classList, "Q", path, inputfeature, mx_json)
        moran_result = moranCal("Q")
        #print(moran_result)
        if(moran_result["moran_p"] > 0.05):
            dict['moran'] = -1
        else:
            dict['moran'] = moran_result['moran_I']
        EI_list.append(dict)
    return EI_list

def getMSD(list, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    dict = {
        'max': max,
        'min': min,
    }
    EI = mapclassify.StdMean(list)
    EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
    dict['breaks'] = EI_bins
    dict['k'] = EI.k
    dict['adcm'] = EI.adcm
    dict['gadf'] = EI.gadf
    dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
    classList = EI.yb.tolist()
    makeGeoJson(classList, "MSD", path, inputfeature, mx_json)
    moran_result = moranCal("MSD")
    #print(moran_result)
    if(moran_result["moran_p"] > 0.05):
        dict['moran'] = -1
    else:
        dict['moran'] = moran_result['moran_I']
    EI_list.append(dict)
    return EI_list

def getMaximumBreaks(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        EI = mapclassify.MaximumBreaks(list, k=i)
        EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
        dict['breaks'] = EI_bins
        dict['adcm'] = EI.adcm
        dict['gadf'] = EI.gadf
        dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
        classList = EI.yb.tolist()
        makeGeoJson(classList, "MB", path, inputfeature, mx_json)
        moran_result = moranCal("MB")
        #print(moran_result)
        if(moran_result["moran_p"] > 0.05):
            dict['moran'] = -1
        else:
            dict['moran'] = moran_result['moran_I']
        EI_list.append(dict)
    return EI_list

def getHeadTailBreaks(list, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    dict = {
        'max': max,
        'min': min,
    }
    EI = mapclassify.HeadTailBreaks(list)
    EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
    dict['breaks'] = EI_bins
    dict['k'] = EI.k
    dict['adcm'] = EI.adcm
    dict['gadf'] = EI.gadf
    dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
    classList = EI.yb.tolist()
    makeGeoJson(classList, "HT", path, inputfeature, mx_json)
    moran_result = moranCal("HT")
    #print(moran_result)
    if(moran_result["moran_p"] > 0.05):
        dict['moran'] = -1
    else:
        dict['moran'] = moran_result['moran_I']
    EI_list.append(dict)
    return EI_list

def getJenksCaspall(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        EI = mapclassify.JenksCaspall(list, k=i)
        EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
        dict['breaks'] = EI_bins
        dict['adcm'] = EI.adcm
        dict['gadf'] = EI.gadf
        dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
        classList = EI.yb.tolist()
        makeGeoJson(classList, "JC", path, inputfeature, mx_json)
        moran_result = moranCal("JC")
        #print(moran_result)
        if(moran_result["moran_p"] > 0.05):
            dict['moran'] = -1
        else:
            dict['moran'] = moran_result['moran_I']
        EI_list.append(dict)
    return EI_list

def getFisherJenks(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        EI = mapclassify.FisherJenks(list, k=i)
        EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
        dict['breaks'] = EI_bins
        dict['adcm'] = EI.adcm
        dict['gadf'] = EI.gadf
        dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
        classList = EI.yb.tolist()
        makeGeoJson(classList, "FJ", path, inputfeature, mx_json)
        moran_result = moranCal("FJ")
        #print(moran_result)
        if(moran_result["moran_p"] > 0.05):
            dict['moran'] = -1
        else:
            dict['moran'] = moran_result['moran_I']
        EI_list.append(dict)
    return EI_list

def getMaxp(list, init_k, max_k, path, inputfeature, mx_json):
    EI_list = []
    max = np.max(list)
    min = np.min(list)
    for i in range(init_k, max_k+1):
        dict = {
            'k': i,
            'max': max,
            'min': min,
        }
        try:
            EI = mapclassify.MaxP(list, k=i)
            EI_bins = np.delete(EI.bins, -1).tolist() # remove the upperbound
            dict['breaks'] = EI_bins
            dict['adcm'] = EI.adcm
            dict['gadf'] = EI.gadf
            dict['GVF'] = goodness_of_variance_fit(np.array(list),EI.bins)
            classList = EI.yb.tolist()
            makeGeoJson(classList, "MP", path, inputfeature, mx_json)
            moran_result = moranCal("MP")
            #print(moran_result)
            if(moran_result["moran_p"] > 0.05):
                dict['moran'] = -1
            else:
                dict['moran'] = moran_result['moran_I']
            EI_list.append(dict)
        except:
            print("error at:" + str(i))
    return EI_list

def moranCal(classificationName):
    shp = geopandas.read_file("temp.shp")
    shp_filtered = shp[isNaN(shp[classificationName])]
    #print(shp_filtered[classificationName])
    weights = Queen.from_dataframe(shp_filtered)
    moran = Moran(shp_filtered[classificationName], weights)
    moran_result = {
        'moran_EI': round(moran.EI,2),
        'moran_I': round(moran.I,2),
        'moran_p': round(moran.p_norm,2)
    }

    return moran_result

def makeGeoJson(classList, classificationName, path, inputfeature, mx_json):
    rawJson = json.loads(mx_json)
    j = 0
    for feature in rawJson["features"]:
        if feature['properties'][inputfeature] != "unknown":
            #print(feature['properties']["STATE_NAME"])
            feature['properties'][classificationName] = classList[j]
        j = j + 1

    with open('temp.json', 'w', encoding='utf-8') as f:
        json.dump(rawJson, f, ensure_ascii=False)

    jsonObj = geopandas.read_file("temp.json")
    jsonObj.to_file('temp.shp')
    print("temp.shp generated!")

def makeOutPut(path, k_init, k_max, feature):
    output_json = {}
    mx = geopandas.read_file(path)
    mx_filtered = mx[~mx.STATE_NAME.isin(['United States Virgin Islands', 'Puerto Rico', 'Alaska', 'Hawaii'])]
    mx_json = mx_filtered.to_json()
    featureList = mx_filtered[feature].tolist()
    featureList = [round(float(e), 2) for e in featureList]
    # formating output file
    output_json['description'] = feature
    output_json['classification_methods'] = [
        "equal_interval",
        "quantile",
        "mean_standard_deviation",
        "maximum_breaks",
        "head_tail",
        "jenks_caspall",
        "fisher_jenks",
        "max_p"
    ]
    output_json['classification_methods_title'] = [
        "Equal interval",
        "Quantile",
        "Mean Standard Deviation",
        "Maximum breaks",
        "Head tail",
        "Jenks Caspall",
        "Fisher Jenks",
        "Max p"
    ]
    output_json["key_prop"] = feature
    output_json['data_list'] = featureList

    # get Equal Interval result
    '''
    Equal intervals have the dual advantages of simplicity and ease of interpretation. 
    However, this rule only considers the extreme values of the distribution and, 
    in some cases, this can result in one or more classes being sparse.
    '''
    EI_dict = getEqualInterval(featureList, k_init, k_max, path, feature, mx_json)
    output_json['equal_interval'] = EI_dict
    #print(EI_dict)

    # get quantile result
    '''
    The varying widths of the intervals can be markedly different which
    can lead to problems of interpretation. A second challenge facing quantiles
    arises when there are a large number of duplicate values in the distribution
    such that the limits for one or more classes become ambiguous.
    '''
    Quantile_dict = getQuantiles(featureList, k_init, k_max, path, feature, mx_json)
    output_json['quantile'] = Quantile_dict

    # 2. Fisher Jenks
    FJ = getFisherJenks(featureList, k_init, k_max, path, feature, mx_json)
    output_json['fisher_jenks'] = FJ

    with open('vis24/state_bachelors_or_higher_25up_feature.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False)
    print("Done!!!")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    makeOutPut("vis24/state456.json", 2, 7, "bachelors_or_higher_25up")
