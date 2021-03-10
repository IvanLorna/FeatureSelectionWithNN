%CS170 Project 2 Live Script File
addpath('E:\Winter2021\CS170\Project2\FeatureSelectionWithNN')

%read data from file
%data = readmatrix('CS170_largetestdata__2.txt');
data = readmatrix('CS170_SMALLtestdata__42.txt');

disp(['Read file of ',num2str(size(data,1)),' data points with ',num2str(size(data,2)),' features each.']);

%find best performing combination of features through Search
[best_features , ordered_features, performances]= Feature_Search(data)
%%
function [best_features, ordered_features, performances] = Feature_Search(data)
    best_feature_accuracies = [];
    
    current_set_of_features = [];
    for i = 1:size(data,2)
        disp(['On level ',num2str(i),' of search tree']);
        feature_to_add_at_this_level = [];
        best_so_far_accuracy = 0;
        
        for k = 1:size(data,2)
            if isempty(intersect(current_set_of_features,k))
                disp(['---Considering adding feature ', num2str(k)]);
                accuracy = Leave_One_Out_Cross_Validation(data,current_set_of_features,k);
                
                if accuracy > best_so_far_accuracy
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                end
            end
        end
        current_set_of_features(i) = feature_to_add_at_this_level;
        best_feature_accuracies(i) = best_so_far_accuracy;
        disp(['On level ', num2str(i),', added feature ',num2str(feature_to_add_at_this_level),'.']);

    end
    [~,ind] = max(best_feature_accuracies);
    best_features = current_set_of_features(1:ind);
    ordered_features = current_set_of_features;
    performances = best_feature_accuracies;
end

function accuracy = Leave_One_Out_Cross_Validation(data,current_set_of_features,k)
%edit to change cross validation group size    
cross_cut = 5;
    %TO ASK AT OFFICE HOURS: how to cut data into partitions nicely in
    %matlab
    accuracy = rand;
end