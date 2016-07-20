#pragma once

#include <vector>

#include <random>

class GSDR {
public:
    struct Node {
        std::vector<float> _weights;
        
        float _activation;
        float _state;

        float _bias;
    };

private:
    std::vector<Node> _SDRNodes;
    std::vector<Node> _forcedLatentNodes;

public:
    float _activeRatio;

    GSDR()
        : _activeRatio(0.1f)
    {}

    void createRandom(int numInputs, int numHidden, int numForcedLatents, float initMinWeight, float initMaxWeight, float forceLatentInfluence, std::mt19937 &generator);

    void learn(const std::vector<float> &inputs, const std::vector<float> &forcedLatents, float alpha, float beta);

    void generate(const std::vector<float> &forcedLatents, std::vector<float> &generation);
};