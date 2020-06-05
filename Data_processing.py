import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

data = {'data_file': [str(i).zfill(2) for i in range(1,61) ], 'Num_of_upperlanes': [2,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3.5,3.5,3.5]}
lane_data  = pd.DataFrame(data)
two_lane_data = lane_data[lane_data['Num_of_upperlanes']==2].set_index('data_file').T


#iterate through the folder to give total number
def num_lanechanges(folder_path):
    
    files = [file_path for file_path in os.listdir(folder_path) if file_path.endswith('.csv')]
    df = pd.read_csv(os.path.join(folder_path, files[0]), )
    left = df[(df['drivingDirection'] == 1)&(df['numLaneChanges'] > 0)].numLaneChanges.count()
    right = df[(df['drivingDirection'] == 2)&(df['numLaneChanges'] > 0)].numLaneChanges.count()
    total = left+right
    old_df = pd.DataFrame([left, right, total])
        
    for file_path in files[1:]:
        df1 = pd.read_csv(os.path.join(folder_path, file_path))
        left = df1[(df1['drivingDirection'] == 1)&(df1['numLaneChanges'] > 0)].numLaneChanges.count()
        right = df1[(df1['drivingDirection'] == 2)&(df1['numLaneChanges'] > 0)].numLaneChanges.count()
        total = left+right
        new_df = pd.DataFrame([left, right, total])
        old_df = pd.concat([old_df, new_df], axis=1)
        
    return old_df 

def lane_change_dataframe(x):
    x.columns = [str(i).zfill(2) for i in range(1,61)]
    x.index = ['left_lane ','right','Total']
    return x


def lane_change_scenarios(df_tracksMeta, df_tracks):
    df_extracted_meta = df_tracksMeta[df_tracksMeta['numLaneChanges'] == 1]
    df_extracted_meta = df_extracted_meta.iloc[:, 0:6]
    df_lane_change_id = np.array(df_extracted_meta['id'])
    df_extracted_tracks = df_tracks[df_tracks.id.isin(df_lane_change_id)]
    df_extracted_tracks =  df_extracted_tracks.drop(axis=1, labels = list_column_drop_variables ) 
    return df_extracted_meta, df_lane_change_id , df_extracted_tracks

def ego_vehicle_df(vehicle_df,start, end, drop):
    new_data = vehicle_df[(vehicle_df['frame'] >= start) & (vehicle_df['frame'] <= end )].drop(axis=1, labels = drop)
    new_data = new_data.set_index('frame')
    ego_df = pd.DataFrame(np.nan, index = np.arange(start, end ), columns = ['id','x','y','xVelocity','yVelocity'])
    ego_df.update(new_data)
    return ego_df

def surrounding_vehicle_df(vehicle_following_df,start,end,drop):
    non_ego_vehicle = vehicle_following_df[(vehicle_following_df['frame'] > start) & (vehicle_following_df['frame'] <=end )].drop(axis=1, labels = drop) 
    non_ego_vehicle = non_ego_vehicle.set_index('frame')
    surrounding_df = pd.DataFrame(np.nan, index = np.arange(start, end ), columns = ['id','x','y','xVelocity','yVelocity'])
    surrounding_df.update(non_ego_vehicle)
    return surrounding_df

def fill_nan(dataframe, v_id):
    dataframe['id'] = dataframe.id.fillna(v_id)
    dataframe = dataframe.interpolate(method ='linear', limit_direction ='backward', limit = 25)
    #dataframe = dataframe.fillna(pd.rolling_mean(dataframe, 20, min_periods=2)).round(2)
    dataframe = dataframe.rolling(window = 20, min_periods=1).mean().round(2)
    return dataframe

def mirror_lower_data(x):
    if(x >= 24.96):
        a = 24.96 - abs(x-24.96)
    else:
        a = 24.96 + abs(24.99-x) 
    return a

def mirror_lower_data_follow(x):
    a = 24.96 + abs(x-24.96)
    return a
          
    
def mirror_upper_data(x):
    if(x >= 12.59):
        a = 12.59 - abs(x-12.59)
    else:
        a = 12.59 + abs(12.59-x) 
    return a

def mirror_upper_data_follow(x):
    a = 12.59 + abs(x -12.59)
    return a


def shift_lower_to_upper_data(x):
    a = x - 12.37
    return a
    
def change_sign(x):
    if (x < 0):
        x = abs(x)
    elif( x==0):
        x = x
    else:
        x = -x
    return x

def complete_data_frame(df):
    df[10] = df[1] - df[6] #distance x position
    df[11] = abs(df[2] - df[7]) #y postion difference
    df[12] = df[3] - df[8] #relative x velocity
    df[13] = df[4] - df[9] #relative y velocity
    return df


def lane_change_data(df):
    
    final_data = pd.DataFrame()
    final_odd_data = pd.DataFrame()
    final_even_data = pd.DataFrame()
   
    for i in range(len(lanechange_id)):

        #vehicle id of lane change
        vehicle_id = lanechange_id[i]
        vehicle_df = df[df['id'] == vehicle_id]

        #lane in  which vehicle is travelling
        start_lane = int(df[df['id']== vehicle_id].laneId.head(1))

        #choose the frame befor and after the lane change
        frame = int(df[(df['id'] == vehicle_id) & (df['laneId'] == start_lane)].frame.tail(1))


        # return frame
        intial_state, final_state, next_lane_state =  frame-119 , frame + 51,  frame +1

        #intial/ final frame seen
        first_seen = int(df_tracksMeta[df_tracksMeta['id'] == vehicle_id].initialFrame)
        last_seen = int(df_tracksMeta[df_tracksMeta['id'] == vehicle_id].finalFrame)

        #intial state(frame) should genrealy be greater than the first seen
        difference_start  = intial_state - first_seen
        difference_end = final_state - last_seen  


        odd = list(range(1,170+1, 2))
        even = list(range(0,170, 2))
        ###############################################################################
        if(start_lane == 2):


            if difference_start  >  -20 and difference_end < 13:

                #left data is the ego vehicle data set
                left_data_set = ego_vehicle_df(vehicle_df,intial_state, final_state ,drop_list)

                #check the number of null items should be  less than 20
                left_null = left_data_set.x.isnull().sum()
                if (left_null < 20):

                    #mirror the  ego vehicle data along the x axis similar to lower lane changes
                    left_data_set['x'] = left_data_set.x.apply(lambda x: -410+x).apply(lambda x: abs(x))
                    left_data_set['xVelocity'] = left_data_set.xVelocity.apply(lambda x: abs(x))

                    #impute the data with interpolate and fillna
                    left_data_set_imputed = fill_nan(left_data_set,vehicle_id)

                    #shift the data to the centre of the road
                    difference = float(left_data_set_imputed.x.head(1).round(2))
                    shift_centre = 210 - difference 

                    left_data_set_imputed ['x'] = left_data_set_imputed.x.apply(lambda x: x + shift_centre)



                else:
                    continue



                #id of lane following
                vehicle_following_id = int(vehicle_df[vehicle_df['frame'] == next_lane_state].followingId)

                #continue only if the following vehicle is seen during the lane change scenarios
                if (vehicle_following_id != 0):

                    #actual vehicle following data
                    vehicle_following_df = df_surrounding[df_surrounding['id'] == vehicle_following_id]
                    distance_x = int(vehicle_df[vehicle_df['frame']== next_lane_state].x)
                    disatance_y = int(vehicle_following_df[vehicle_following_df['frame'] == next_lane_state].x)
                    theta = distance_x - disatance_y #distance between the ego and following 

                    #continue if the distance between the ego and follow is less than 100 or 150
                    if theta < 125:

                        #vehicle follwing data set with correct length with appropriate frame 
                        right_data_set = surrounding_vehicle_df(vehicle_following_df,intial_state, final_state, drop_list)
                        right_null = right_data_set.x.isnull().sum()

                        if right_null < 20 :

                            #mirror the  ego vehicle data along the y axis similar to lower lane changes
                            right_data_set['x'] = right_data_set.x.apply(lambda x: -410+x).apply(lambda x: abs(x))
                            right_data_set['xVelocity'] = right_data_set.xVelocity.apply(lambda x: abs(x))

                            #impute the data with interpolate and fillna
                            right_data_set_imputed = fill_nan(right_data_set,vehicle_following_id)

                            #shift the data according to the ego data
                            right_data_set_imputed['x'] = right_data_set_imputed.x.apply(lambda x: x + shift_centre)


                            #data frame of containg the ego and follow 
                            data = pd.concat([left_data_set_imputed, right_data_set_imputed],axis = 1, ignore_index=True)

                            #dataframe - ego, follow, diffrence
                            data = complete_data_frame(data)

                            #divide the data set into odd and even
                            odd_data = data.take(odd)
                            even_data = data.take(even)

                        else:
                            continue
                    else:
                        continue

                else:
                    continue

            else:
                continue     

        ############################################################################################
        #condition if the start lane is 3
        elif (start_lane == 3):
            if difference_start  >  -20 and difference_end < 13:
                left_data_set = ego_vehicle_df(vehicle_df,intial_state, final_state ,drop_list)
                left_null = left_data_set.x.isnull().sum()

                if (left_null < 20):

                    ## process data as to match the vehicle flowing in postive x similar to lower lane changes
                    left_data_set['x'] = left_data_set.x.apply(lambda x: -410 + x).apply(lambda x: abs(x))
                    left_data_set['xVelocity'] =left_data_set.xVelocity.apply(lambda x: abs(x))

                    #impute the data with interpolate and fillna
                    left_data_set_imputed = fill_nan(left_data_set,vehicle_id)

                    #shift the data to the centre of the road
                    difference = float(left_data_set_imputed.x.head(1).round(2))
                    shift_centre = 210 - difference  
                    left_data_set_imputed ['x'] = left_data_set_imputed.x.apply(lambda x: x + shift_centre)

                    #mirrior along the lane marking, making lane change along the postive lane change
                    left_data_set_imputed['y'] = left_data_set_imputed.y.apply(mirror_upper_data)
                    left_data_set_imputed['yVelocity'] = left_data_set_imputed.yVelocity.apply(change_sign)

                else:
                    continue

                #id of lane following
                vehicle_following_id = int(vehicle_df[vehicle_df['frame'] == next_lane_state].followingId)

                if (vehicle_following_id != 0):
                    vehicle_following_df = df_surrounding[df_surrounding['id'] == vehicle_following_id]

                    distance_x = int(vehicle_df[vehicle_df['frame']== next_lane_state].x)
                    disatance_y = int(vehicle_following_df[vehicle_following_df['frame'] == next_lane_state].x)
                    theta = distance_x - disatance_y #distance between the ego and following 


                    if theta < 125:                                                                    
                        right_data_set = surrounding_vehicle_df(vehicle_following_df,intial_state, final_state, drop_list)
                        right_null = right_data_set.x.isnull().sum()

                        if right_null < 20 :

                            # process data as to match the vehicle flowing in postive x similar to lower lane changes
                            right_data_set['x'] = right_data_set.x.apply(lambda x: -410 + x).apply(lambda x: abs(x))
                            right_data_set['xVelocity'] =right_data_set.xVelocity.apply(lambda x: abs(x))

                            #impute the data with interpolate and fillna
                            right_data_set_imputed = fill_nan(right_data_set,vehicle_following_id)

                            #shift the data according to the ego data
                            right_data_set_imputed['x'] = right_data_set_imputed.x.apply(lambda x: x + shift_centre)

                            #mirrior along the lane marking, making lane change along the postive lane change
                            right_data_set_imputed['y'] = right_data_set_imputed.y.apply(mirror_upper_data_follow)
                            right_data_set_imputed['yVelocity'] = right_data_set_imputed.yVelocity.apply(lambda x: abs(x))

                            #data frame of containg the ego and follow 
                            data= pd.concat([left_data_set_imputed, right_data_set_imputed],axis = 1, ignore_index=True)

                            #dataframe - ego, follow, diffrence
                            data = complete_data_frame(data)

                            #divide the data set into odd and even
                            odd_data = data.take(odd)
                            even_data = data.take(even)


                        else:
                            continue


                    else:
                        continue

                else:
                    continue

            else:
                continue

         #####################################################################################################
        #lower lane
        elif(start_lane == 5):

            if difference_start  >  -20 and difference_end < 13:

                #left data is the ego vehicle data set
                left_data_set = ego_vehicle_df(vehicle_df,intial_state, final_state ,drop_list)

                #check the number of null items should be  less than 20
                left_null = left_data_set.x.isnull().sum()

                if (left_null < 20):

                    #impute the data with interpolate and fillna
                    left_data_set_imputed = fill_nan(left_data_set,vehicle_id)

                    #shift the data to the centre of the road
                    difference = float(left_data_set_imputed.x.head(1).round(2))
                    shift_centre = 210 - difference 
                    left_data_set_imputed ['x'] = left_data_set_imputed.x.apply(lambda x: x + shift_centre)

                    #move the data to upper lane 
                    left_data_set_imputed['y'] = left_data_set_imputed.y.apply(shift_lower_to_upper_data)


                else:
                    continue



                #id of lane following
                vehicle_following_id = int(vehicle_df[vehicle_df['frame'] == next_lane_state].followingId)


                if (vehicle_following_id != 0):

                    vehicle_following_df = df_surrounding[df_surrounding['id'] == vehicle_following_id]
                    distance_x = int(vehicle_df[vehicle_df['frame']== next_lane_state].x)
                    disatance_y = int(vehicle_following_df[vehicle_following_df['frame'] == next_lane_state].x)
                    theta = distance_x - disatance_y #distance between the ego and following 


                    if theta < 125:

                        #vehicle follwing data set with correct length with appropriate frame 
                        right_data_set = surrounding_vehicle_df(vehicle_following_df,intial_state, final_state, drop_list)
                        right_null = right_data_set.x.isnull().sum()


                        if right_null < 20 :
                            right_data_set_imputed = fill_nan(right_data_set,vehicle_following_id)

                            #shift the data according to the ego data
                            right_data_set_imputed['x'] = right_data_set_imputed.x.apply(lambda x: x + shift_centre)

                            #move the data to upper lane 
                            right_data_set_imputed['y'] = right_data_set_imputed.y.apply(shift_lower_to_upper_data)

                            #data frame of containg the ego and follow 
                            data= pd.concat([left_data_set_imputed, right_data_set_imputed],axis = 1, ignore_index=True )

                            #dataframe - ego, follow, diffrence
                            data = complete_data_frame(data)

                            #divide the data set into odd and even
                            odd_data = data.take(odd)
                            even_data = data.take(even)

                        else:
                            continue
                    else:
                        continue

                else:
                    continue

            else:
                continue     

        #########################################################################################
        else:

            if difference_start  >  -20 and difference_end < 13:

                #left data is the ego vehicle data set
                left_data_set = ego_vehicle_df(vehicle_df,intial_state, final_state ,drop_list)

                #check the number of null items should be  less than 20
                left_null = left_data_set.x.isnull().sum()

                if (left_null < 20):

                    #impute the data with interpolate and fillna
                    left_data_set_imputed = fill_nan(left_data_set,vehicle_id)

                    #shift the data to the centre of the road
                    difference = float(left_data_set_imputed.x.head(1).round(2))
                    shift_centre = 210 - difference  
                    left_data_set_imputed ['x'] = left_data_set_imputed.x.apply(lambda x: x + shift_centre)

                    #mirror the data along lane marking
                    left_data_set_imputed['y'] = left_data_set_imputed.y.apply(mirror_lower_data)
                    left_data_set_imputed['yVelocity'] = left_data_set_imputed.yVelocity.apply(change_sign)

                    #shift the data to upper lane
                    left_data_set_imputed['y'] = left_data_set_imputed.y.apply(shift_lower_to_upper_data)




                else:
                    continue


                #id of lane following
                vehicle_following_id = int(vehicle_df[vehicle_df['frame'] == next_lane_state].followingId)

                if (vehicle_following_id != 0):

                    #actual vehicle following data
                    vehicle_following_df = df_surrounding[df_surrounding['id'] == vehicle_following_id]
                    distance_x = int(vehicle_df[vehicle_df['frame']== next_lane_state].x)
                    disatance_y = int(vehicle_following_df[vehicle_following_df['frame'] == next_lane_state].x)

                    theta = distance_x - disatance_y #distance between the ego and following 

                    #continue if the distance between the ego and follow is less than 100 or 150
                    if theta < 125:

                        #actual follow vehicle data
                        right_data_set = surrounding_vehicle_df(vehicle_following_df,intial_state, final_state, drop_list)

                        #check the number of null items should be  less than 20
                        right_null = right_data_set.x.isnull().sum()                                                                       
                        if right_null < 20 :   

                            #impute the data with interpolate and fillna
                            right_data_set_imputed = fill_nan(right_data_set,vehicle_following_id)

                            #shift the data according to the ego data
                            right_data_set_imputed['x'] = right_data_set_imputed.x.apply(lambda x: x + shift_centre)

                            #mirror the data along lane marking
                            right_data_set_imputed['y'] = right_data_set_imputed.y.apply(mirror_lower_data_follow)
                            right_data_set_imputed['yVelocity'] = right_data_set_imputed.yVelocity.apply(change_sign)

                            #move the data to upper lane
                            right_data_set_imputed['y'] = right_data_set_imputed.y.apply(shift_lower_to_upper_data)


                            #data frame of containg the ego and follow 
                            data= pd.concat([left_data_set_imputed, right_data_set_imputed],axis = 1, ignore_index=True)

                            #dataframe - ego, follow, diffrence
                            data = complete_data_frame(data)

                            #divide the data set into odd and even
                            odd_data = data.take(odd)
                            even_data = data.take(even)

                        else:
                            continue


                    else:
                        continue

                else:
                    continue

            else:
                continue

        ############################################################      
        final_data = final_data.append(data, ignore_index=True)
        final_odd_data = final_odd_data.append(odd_data,ignore_index=True)
        final_even_data = final_even_data.append(even_data, ignore_index=True)
        ############################################################

    final_data.columns = ['ego', 'ego_x', 'ego_y','ego_xVelocity', 'ego_yVelocity','follow', 'follow_x', 'follow_y','follow_xVelocity', 'follow_yVelocity','theta_x', 'theta_y','theta_xVelocity', 'theta_yVelocity']
    final_odd_data.columns = ['ego', 'ego_x', 'ego_y','ego_xVelocity', 'ego_yVelocity','follow', 'follow_x', 'follow_y','follow_xVelocity', 'follow_yVelocity','theta_x', 'theta_y','theta_xVelocity', 'theta_yVelocity']    
    final_even_data.columns = ['ego', 'ego_x', 'ego_y','ego_xVelocity', 'ego_yVelocity','follow', 'follow_x', 'follow_y','follow_xVelocity', 'follow_yVelocity','theta_x', 'theta_y','theta_xVelocity', 'theta_yVelocity']    
    
    return final_even_data, final_odd_data


two_lanedata_list = list(two_lane_data.columns)
Trainset,Testset = two_lanedata_list[0:9], two_lanedata_list[9:]

lane_change = num_lanechanges('C:/Users/adity/Thesis project/data/tracksMeta')
lanechange_df = lane_change_dataframe(lane_change)

extracted lane change data
df_tracksMeta = pd.read_csv("C:/Users/adity/Thesis project/data/tracksMeta/01_tracksMeta.csv")
df_tracks = pd.read_csv(("C:/Users/adity/Thesis project/data/tracks/01_tracks.csv"))


list_column_drop_variables = ['width','height','xAcceleration', 'yAcceleration','frontSightDistance','backSightDistance','dhw','thw','ttc',
                             'precedingXVelocity','precedingId','leftPrecedingId','leftAlongsideId','rightPrecedingId','rightAlongsideId']

lanechange_tracksMeta ,lanechange_id, lanechange_tracks  = lane_change_scenarios(df_tracksMeta, df_tracks) 

df =  lanechange_tracks 
df_surrounding  = df_tracks.drop(axis=1, labels = list_column_drop_variables)
drop_list = ['followingId','leftFollowingId','rightFollowingId','laneId']
even , odd = lane_change_data(df)

for i in range(len(a)):
    data = odd.groupby('ego').get_group(a[i])
    log_p_data1 = data[['ego_x' ,'follow_x','theta_x']].T.values 
    lat_p_data1 = data[['ego_y' ,'follow_y','theta_y']].T.values 
    log_v_data1 = data[['ego_xVelocity','follow_xVelocity','theta_xVelocity']].T.values
    lat_v_data1 = data[['ego_yVelocity', 'follow_yVelocity','follow_yVelocity']].T.values
    log_p_data = np.dstack((log_p_data, log_p_data1))
    lat_p_data = np.dstack((lat_p_data, lat_p_data1))
    log_v_data = np.dstack((log_v_data, log_v_data1))
    lat_v_data = np.dstack((lat_v_data, lat_v_data1))



np.save(os.path.join('C:/Users/adity/Thesis project/', 'lat_position'), lat_p_data)
np.save(os.path.join('C:/Users/adity/Thesis project/', 'log_position'), log_p_data)
np.save(os.path.join('C:/Users/adity/Thesis project/', 'logvel_position'), log_v_data)
np.save(os.path.join('C:/Users/adity/Thesis project/', 'latvel_position'), lat_v_data)
