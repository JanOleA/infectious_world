#include <limits>
#include <vector>
#include <string.h>


struct Position {
    int x;
    int y;
};


struct PathFindStruct {
    Position pos;
    Position camefrom;
    double cost = 1;
    double gscore = std::numeric_limits<double>::infinity();
    double fscore = std::numeric_limits<double>::infinity();
    bool in_open_set = false;
};


class Finder {
private:
    std::vector<std::vector<PathFindStruct>> map_grid_clear;
    std::vector<PathFindStruct> reconstruct_path(PathFindStruct target,
                                                 std::vector<std::vector<PathFindStruct>> map_grid);
    void map_from_2d(std::vector<std::vector<int>> map_grid);
    double heuristic(Position pos1, Position pos2);
    bool write_to_file;
    Finder(std::vector<std::vector<PathFindStruct>> map_grid, bool write_to_file);
    Finder(std::vector<std::vector<PathFindStruct>> map_grid);
    std::vector<PathFindStruct> find_path(PathFindStruct start,
                                          PathFindStruct target);
    std::vector<std::vector<PathFindStruct>> get_map_grid();
    PathFindStruct get_node(int x, int y);
public:
    Finder(std::vector<std::vector<int>> map_grid, bool write_to_file_);
    Finder(std::vector<std::vector<int>> map_grid);
    std::vector<std::vector<int>> find_path(std::vector<int> start,
                                            std::vector<int> target);
};