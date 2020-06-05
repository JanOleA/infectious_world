%module pathfinding
 %{
 #include "pathfinding.h"
 %}

 %include "std_vector.i"

 namespace std {
    %template(VectorInt) vector<int>;
    %template(VectorVectorInt) vector<vector<int>>;
 };

 %naturalvar Finder::map_grid;
 %naturalvar Finder::map_grid_clear;
 %naturalvar Finder::start_in;
 %naturalvar Finder::target_in;
 %include "pathfinding.h"

