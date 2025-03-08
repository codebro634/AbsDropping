
#include "../../../include/Games/MDPs/RaceTrack.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <queue>
#include <set>
using namespace std;

using namespace RT;

Model::Model(const std::string& fileName, double fail_prob, bool reset_at_crash) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return;
    }

    int rows, cols;
    std::string line;

    // Read the first line for dimensions
    if (std::getline(file, line)) {
        sscanf(line.c_str(), "dim: %d %d", &rows, &cols);
    }

    // Initialize obstacle_map and goal_map with false values
    obstacle_map.resize(rows, std::vector<bool>(cols, false));
    goal_map.resize(rows, std::vector<bool>(cols, false));

    // Parse the file content row by row
    for (int r = 0; r < rows; ++r) {
        if (std::getline(file, line)) {
            for (int c = 0; c < cols; ++c) {
                char ch = line[c];
                if (ch == 'x') {
                    obstacle_map[r][c] = true;
                } else if (ch == 's') {
                    start_positions.push_back({r, c});
                } else if (ch == 'g') {
                    goal_map[r][c] = true;
                }
            }
        }
    }
    file.close();

    calculate_goal_distances();

    this->fail_prob = fail_prob;
    this->reset_at_crash = reset_at_crash;
}

double Model::getDistance(const ABS::Gamestate* a, const ABS::Gamestate* b) const {
    const Gamestate* state_a = (Gamestate*) a;
    const Gamestate* state_b = (Gamestate*) b;
    return abs(state_a->x - state_b->x) + abs(state_a->y - state_b->y) + abs(state_a->dx - state_b->dx) + abs(state_a->dy - state_b->dy);
}

void Model::calculate_goal_distances() {
    set<std::pair<int,int>> goal_positions;
    for(size_t i = 0; i < goal_map.size(); i++) {
        for(size_t j = 0; j < goal_map[0].size(); j++) {
            if(goal_map[i][j]) {
                goal_positions.insert({i,j});
            }
        }
    }

    for(size_t i = 0; i < obstacle_map.size(); i++) {
        for(size_t j = 0; j < obstacle_map[0].size(); j++) {
            if(valid_pos(i,j)) {
                auto start = std::make_pair(i,j);

                // Directions: up, down, left, right
                std::vector<std::pair<int,int>> directions = {
                    {-1, 0}, // Up
                    {1, 0},  // Down
                    {0, -1}, // Left
                    {0, 1}   // Right
                };

                std::vector<std::vector<bool>> visited(obstacle_map.size(), std::vector<bool>(obstacle_map[0].size(), false));

                // Queue for BFS
                queue<std::pair<std::pair<int,int>,int>> q;
                q.push({start,0});
                visited[start.first][start.second] = true;

                while (!q.empty()) {
                    auto tmp = q.front();
                    auto current = tmp.first;
                    q.pop();

                    if(goal_positions.contains(current)) {
                        distances_to_goal[start] = tmp.second;
                        break;
                    }

                    for (const auto& dir : directions) {
                        int new_x = current.first + dir.first;
                        int new_y = current.second + dir.second;
                        if (valid_pos(new_x, new_y) && !visited[new_x][new_y]) {
                            q.push({{new_x, new_y},tmp.second+1});
                            visited[new_x][new_y] = true;
                        }
                    }
                }

                if(!distances_to_goal.contains(start))
                    throw std::runtime_error("No path found to goal from valid start position");
            }
        }
    }

}


double Model::heuristicsValue(ABS::Gamestate* state) const {
    return - distances_to_goal.at({dynamic_cast<RT::Gamestate*>(state)->x, dynamic_cast<RT::Gamestate*>(state)->y});
}

bool Gamestate::operator==(const ABS::Gamestate& other) const
{
    auto* other_state = dynamic_cast<const Gamestate*>(&other);
    return x == other_state->x && y == other_state->y && dx == other_state->dx && dy == other_state->dy && terminal == other_state->terminal;
}

size_t Gamestate::hash() const
{
    int sign_dx = dx < 0? 1: 0;
    int sign_dy = dy < 0? 1: 0;
    return (static_cast<size_t> (x) | (y << 10) | (std::abs(dx) << 20) | (std::abs(dy) << 25) | (sign_dx << 30) | (sign_dy << 31)); //reserve more bits for position
}

void Model::printState(ABS::Gamestate* state) {
    auto* rtState = dynamic_cast<RT::Gamestate*>(state);
    std::cout << "x: " << rtState->x << " y: " << rtState->y << " dx: " << rtState->dx << " dy: " << rtState->dy << std::endl;
    //print the obstalce map with car in it
    for (size_t i = 0; i < obstacle_map[0].size(); i++) {
        for (size_t j = 0; j < obstacle_map.size(); j++) {
            if (obstacle_map[j][i])
                std::cout << "x";
            else if (goal_map[j][i])
                std::cout << "g";
            else if (static_cast<int>(j) == rtState->x && static_cast<int>(i) == rtState->y)
                std::cout << "C";
            else
                std::cout << ".";
        }
        std::cout << std::endl;
    }
}

void Model::resetToStart(Gamestate* state, std::mt19937& rng) const {
    std::uniform_int_distribution<int> dist(0, start_positions.size() - 1);
    auto start_pos = start_positions[dist(rng)];
    state->x = start_pos.first;
    state->y = start_pos.second;
    state->dx = 0;
    state->dy = 0;
}

ABS::Gamestate* Model::getInitialState(std::mt19937& rng)  {
    auto* state = new RT::Gamestate();
    resetToStart(state, rng);
    return state;
}

int Model::getNumPlayers() {
    return 1;
}


ABS::Gamestate* Model::copyState(ABS::Gamestate* uncasted_state) {
    auto state = dynamic_cast<Gamestate*>(uncasted_state);
    auto new_state = new Gamestate();
    *new_state = *state; //default copy constructor should work
    return new_state;
}

std::vector<int> Model::getActions_(ABS::Gamestate* uncasted_state)  {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8};
}


//Slightly modified from https://github.com/dair-iitd/oga-uct/blob/master/OGA/race/parsing.h
bool Model::valid_pos(int x, int y) const {
    return (x >= 0) && (x < static_cast<int>(obstacle_map.size())) && (y >= 0) && (y < static_cast<int>(obstacle_map[0].size())) && !obstacle_map[x][y];
}

//Slightly modified from https://github.com/dair-iitd/oga-uct/blob/master/OGA/race/parsing.h
std::pair<int,int> Model::path_interrupt_pos(int x1, int y1, int x2, int y2) {
    assert(valid_pos(x1, y1));
    int rv = 0, ix = 0, lx = 0, iy = 0, ly = 0;

    if( (x1 != x2) && (y1 != y2) ) {
        float m = (float)(y2 - y1) / (float)(x2 - x1);
        float b = (float)(y1*x2 - y2*x1) / (float)(x2 - x1);
        int inc = (x2 - x1 > 0 ? 1 : -1);
        for( float x = x1; (x != x2 + inc) && (rv == 0); x += inc ) {
            float y = m*x + b;
            lx = ix;
            ly = iy;
            ix = (int)x;
            iy = (int)floor(y + 0.5);
            if( !valid_pos(ix, iy) )
                rv = 1;
            else if( goal_map[ix][iy] )
                rv = 2;
        }
    } else if( (x1 != x2) && (y1 == y2) ) {
        ly = iy = (int)y1;
        int inc = (x2 - x1 > 0 ? 1 : -1);
        for( float x = x1; (x != x2 + inc) && (rv == 0); x += inc ) {
            lx = ix;
            ix = (int)x;
            if( !valid_pos(ix, iy) )
                rv = 1;
            else if( goal_map[ix][iy] )
                rv = 2;
        }
    } else if( (x1 == x2) && (y1 != y2) ) {
        lx = ix = (int)x1;
        int inc = (y2 - y1 > 0 ? 1 : -1);
        for( float y = y1; (y != y2 + inc) && (rv == 0); y += inc ) {
            ly = iy;
            iy = (int)y;
            if( !valid_pos(ix, iy) )
                rv = 1;
            else if( goal_map[ix][iy] )
                rv = 2;
        }
    }

    if( rv == 1 )
        return {lx,ly};
    else if (rv == 2) {
        return {ix,iy};
    }else {
        return {x2,y2};
    }
}

std::pair<std::vector<double>,double> Model::applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) {
        auto* state = dynamic_cast<RT::Gamestate*>(uncasted_state);

        if (goal_map[state->x][state->y])
            state->terminal = true;

        //Calculate new acceleration and pos
        bool action_failed = std::uniform_real_distribution<double>(0, 1)(rng) < fail_prob;
        int dx_no_fail = state->dx + (action / 3) - 1;
        int dy_no_fail = state->dy + (action % 3) - 1;
        auto [nx_no_fail,ny_no_fail] = path_interrupt_pos(state->x, state->y, state->x + dx_no_fail, state->y + dy_no_fail);
        auto [nx_fail, ny_fail] = path_interrupt_pos(state->x, state->y, state->x + state->dx, state->y + state->dy);

        bool diff_pos = nx_no_fail != nx_fail || ny_no_fail != ny_fail;
        int dx = action_failed? state->dx : dx_no_fail;
        int dy = action_failed? state->dy : dy_no_fail;
        int nx = action_failed? nx_fail : nx_no_fail;
        int ny = action_failed? ny_fail : ny_no_fail;

        //update acceleration
        state->dx = dx;
        state->dy = dy;
        if (nx != state->x + dx || ny != state->y + dy) {
            if (reset_at_crash)
                resetToStart(state, rng);
            else{
                state->dx = 0;
                state->dy = 0;
            }
        }
        if(goal_map[nx][ny]) {
            state->terminal = true;
            state->dx = 0;
            state->dy = 0;
        }
        //update pos
        state->x = nx;
        state->y = ny;

        return {  {-1}, diff_pos? (action_failed? fail_prob : 1 - fail_prob) : 1.0};
}