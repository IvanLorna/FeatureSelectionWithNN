%CS170 Project 2 Live Script File

%number of features
data = zeros([2,10])

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
    current_set_of_features(i) = feature_to_add_at_this_level
    best_feature_accuracies(i) = best_so_far_accuracy;
    disp(['On level', num2str(i),', added feature ',num2str(feature_to_add_at_this_level),'.']);
end


%plot(best_feature_accuracies);
%%
function accuracy = Leave_One_Out_Cross_Validation(data,current_set_of_features,k)
    accuracy = rand;
end