#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <math.h>
#include <limits>
#include <string.h>
#include <string>
#include "pathfinding.h"
using namespace std;


Finder::Finder(vector<vector<PathFindStruct>> map_grid, bool write_to_file_) {
    map_grid_clear = map_grid;
    write_to_file = write_to_file_;
}


Finder::Finder(vector<vector<PathFindStruct>> map_grid) {
    map_grid_clear = map_grid;
    write_to_file = false;
}


Finder::Finder(vector<vector<int>> map_grid, bool write_to_file_) {
    /* Generates the map from a 2d vector of costs (0 = inf) */
    map_from_2d(map_grid);
    write_to_file = write_to_file_;
}


Finder::Finder(vector<vector<int>> map_grid) {
    /* Generates the map from a 2d vector of costs (0 = inf) */
    map_from_2d(map_grid);
    write_to_file = false;
}


void Finder::map_from_2d(vector<vector<int>> map_grid) {
    vector<vector<PathFindStruct>> map_grid_new;
    int y = 0;
    int x = 0;

    for (vector<int> in_row : map_grid) {
        vector<PathFindStruct> row;
        for (int val : in_row) {
            Position pos{x, y};
            Position cf{-9, -9};
            PathFindStruct new_item{pos, cf, 1};
            if (val == 0) {
                new_item.cost = numeric_limits<double>::infinity();
            } else {
                new_item.cost = val;
            }
            row.push_back(new_item);
            x++;
        }
        map_grid_new.push_back(row);
        x = 0;
        y++;
    }

    map_grid_clear = map_grid_new;
}


vector<PathFindStruct> Finder::reconstruct_path(PathFindStruct target,
                                                vector<vector<PathFindStruct>> map_grid) {
    vector<PathFindStruct> path = {target};
    PathFindStruct current = target;
    Position current_cf = current.camefrom;
    while (current_cf.x != -9 and current_cf.y != -9) {
        current = map_grid[current_cf.y][current_cf.x];
        current_cf = current.camefrom;
        path.push_back(current);
    }

    reverse(path.begin(),path.end());
    return path;
}


double Finder::heuristic(Position pos1, Position pos2) {
    double x_diff = pos1.x - pos2.x;
    double y_diff = pos1.y - pos2.y;

    double dist = sqrt(x_diff*x_diff + y_diff*y_diff);

    return dist;
}


vector<PathFindStruct> Finder::find_path(PathFindStruct start,
                                         PathFindStruct target) {

    vector<vector<PathFindStruct>> map_grid = map_grid_clear;
    ofstream outfile;
    Position target_pos = target.pos;
    Position start_pos = start.pos;
    start.camefrom.x = -9;
    start.camefrom.y = -9;
    start.gscore = 0;
    start.fscore = heuristic(start_pos, target_pos);

    vector<PathFindStruct> open_set = {start};
    vector<double> open_set_fscores = {start.fscore};
    vector<PathFindStruct> closed_set = {};
    start.in_open_set = true;

    map_grid[start_pos.y][start_pos.x] = start;

    int map_x_size = map_grid[0].size();
    int map_y_size = map_grid.size();
    int argmin = 0;
    int other_x;
    int other_y;

    PathFindStruct current;
    PathFindStruct neighbor;
    Position current_pos;
    Position neighbor_pos;
    double tentative_gscore;
    int index;

    if (write_to_file) {
        outfile.open("output.txt");
    }

    while (open_set.size() > 0) {
        argmin = distance(open_set_fscores.begin(),
                          min_element(open_set_fscores.begin(),
                          open_set_fscores.end()));
        current = open_set[argmin];


        if (outfile.is_open()) {
            for (vector<PathFindStruct> row : map_grid) {
                for (PathFindStruct item : row) {
                    outfile << "[";
                    outfile << item.pos.x << ",";
                    outfile << item.pos.y << ",";
                    outfile << item.camefrom.x << ",";
                    outfile << item.camefrom.y << ",";
                    outfile << item.gscore << ",";
                    outfile << item.fscore << ",";
                    outfile << item.cost << ",";
                    outfile << item.in_open_set << "];";
                }
            }
            outfile << "[" << current.pos.x << "," << current.pos.y << "];";
            outfile << "\n";
        }
        

        if (current.pos.x == target.pos.x && current.pos.y == target.pos.y) {
            //cout << "Target reached" << endl;
            if (outfile.is_open()) {
                outfile.close();
            }
            return reconstruct_path(current, map_grid);
        }

        open_set.erase(open_set.begin() + argmin);
        open_set_fscores.erase(open_set_fscores.begin() + argmin);
        current.in_open_set = false;
        closed_set.push_back(current);

        map_grid[current.pos.y][current.pos.x] = current;

        current_pos = current.pos;
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                other_x = current_pos.x + j;
                other_y = current_pos.y + i;
                if (other_x < 0 || other_y < 0) {
                    continue;
                }
                if (other_x >= map_x_size || other_y >= map_y_size) {
                    continue;
                }

                neighbor = map_grid[other_y][other_x];
                neighbor_pos = neighbor.pos;

                tentative_gscore = (double)current.gscore + sqrt((double)(i*i) + (double)(j*j))*(double)neighbor.cost;
                
                if (tentative_gscore < neighbor.gscore) {
                    neighbor.camefrom.x = current_pos.x;
                    neighbor.camefrom.y = current_pos.y;
                    neighbor.gscore = tentative_gscore;
                    neighbor.fscore = tentative_gscore + heuristic(neighbor_pos, target_pos);

                    index = 0;
                    for (PathFindStruct item : open_set) {
                        if (item.pos.x == neighbor.pos.x && item.pos.y == neighbor.pos.y) {
                            break;
                        }
                        index++;
                    }
                    
                    if (index >= open_set.size()) {
                        open_set.push_back(neighbor);
                        open_set_fscores.push_back(neighbor.fscore);
                        neighbor.in_open_set = true;
                    } else {
                        open_set[index] = neighbor;
                        open_set_fscores[index] = neighbor.fscore;
                    }

                    map_grid[other_y][other_x] = neighbor;
                }
            }
        }
    }
    if (outfile.is_open()) {
        outfile.close();
    }
    return {};
}


vector<vector<int>> Finder::find_path(vector<int> start, vector<int> target) {
    /* Takes two vectors of start position and target position as input, and 
    returns a 2D vector containing the path coordinates */
    
    int start_x = start[0];
    int start_y = start[1];
    int target_x = target[0];
    int target_y = target[1];
    PathFindStruct start_ = map_grid_clear[start_y][start_x];
    PathFindStruct target_ = map_grid_clear[target_y][target_x];

    

    vector<PathFindStruct> temp_path = find_path(start_, target_);

    vector<vector<int>> path;

    for (PathFindStruct item : temp_path) {
        vector<int> temp_pos{item.pos.x, item.pos.y};
        path.push_back(temp_pos);
    }

    return path;
}



vector<vector<PathFindStruct>> Finder::get_map_grid() {
    return map_grid_clear;
}


PathFindStruct Finder::get_node(int x, int y) {
    return map_grid_clear[y][x];
}


int main() {
    return 0;
}