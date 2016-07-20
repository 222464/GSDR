#include "GSDR.h"

void GSDR::createRandom(int numInputs, int numHidden, int numForcedLatents, float initMinWeight, float initMaxWeight, float forceLatentInfluence, std::mt19937 &generator) {
    _SDRNodes.resize(numHidden);
    _forcedLatentNodes.resize(numHidden);

    std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
    std::uniform_real_distribution<float> dist11(-1.0f, 1.0f);

    for (int i = 0; i < numHidden; i++) {
        _SDRNodes[i]._weights.resize(numInputs);

        for (int j = 0; j < numInputs; j++)
            _SDRNodes[i]._weights[j] = weightDist(generator);

        _forcedLatentNodes[i]._weights.resize(numForcedLatents);

        for (int j = 0; j < numForcedLatents; j++)
            _forcedLatentNodes[i]._weights[j] = dist11(generator) * forceLatentInfluence;
    }
}

void GSDR::learn(const std::vector<float> &inputs, const std::vector<float> &forcedLatents, float alpha, float beta) {
    // Find latent biases
    for (int i = 0; i < _forcedLatentNodes.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < forcedLatents.size(); j++) {
            float delta = forcedLatents[j] - _forcedLatentNodes[i]._weights[j];

            sum += -delta * delta;
        }

        _forcedLatentNodes[i]._activation = sum;
    }

    // Activate SDR
    for (int i = 0; i < _SDRNodes.size(); i++) {
        float sum = _SDRNodes[i]._bias;

        for (int j = 0; j < inputs.size(); j++) {
            sum += inputs[j] * _SDRNodes[i]._weights[j];
        }

        _SDRNodes[i]._activation = sum + _forcedLatentNodes[i]._activation; // Add in latent bias
    }

    // Inhibit
    for (int i = 0; i < _SDRNodes.size(); i++) {
        float inhibition = 0.0f;

        for (int j = 0; j < _SDRNodes.size(); j++) {
            if (i == j)
                continue;

            if (_SDRNodes[j]._activation >= _SDRNodes[i]._activation)
                inhibition++;
        }

        _SDRNodes[i]._state = inhibition < _activeRatio * _SDRNodes.size() ? 1.0f : 0.0f;
    }

    // Reconstruct
    std::vector<float> recon(inputs.size(), 0.0f);

    for (int i = 0; i < inputs.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < _SDRNodes.size(); j++) {
            sum += _SDRNodes[j]._weights[i] * _SDRNodes[j]._state;
        }

        recon[i] = sum;
    }

    // Learn reconstruction
    for (int i = 0; i < _SDRNodes.size(); i++) {
        for (int j = 0; j < inputs.size(); j++)
            _SDRNodes[i]._weights[j] += alpha * _SDRNodes[i]._state * (inputs[j] - recon[j]);

        _SDRNodes[i]._bias += beta * -_SDRNodes[i]._activation;
    }
}

void GSDR::generate(const std::vector<float> &forcedLatents, std::vector<float> &generation) {
    // Find latent biases
    for (int i = 0; i < _forcedLatentNodes.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < forcedLatents.size(); j++) {
            float delta = forcedLatents[j] - _forcedLatentNodes[i]._weights[j];

            sum += -delta * delta;
        }

        _forcedLatentNodes[i]._activation = sum;
    }

    // Activate SDR
    for (int i = 0; i < _SDRNodes.size(); i++) {
        _SDRNodes[i]._activation = _forcedLatentNodes[i]._activation + _SDRNodes[i]._bias; // Add in latent bias
    }

    // Inhibit
    for (int i = 0; i < _SDRNodes.size(); i++) {
        float inhibition = 0.0f;

        for (int j = 0; j < _SDRNodes.size(); j++) {
            if (i == j)
                continue;

            if (_SDRNodes[j]._activation >= _SDRNodes[i]._activation)
                inhibition++;
        }

        _SDRNodes[i]._state = inhibition < _activeRatio * _SDRNodes.size() ? 1.0f : 0.0f;
    }

    // Reconstruct
    generation.resize(_SDRNodes.front()._weights.size());

    for (int i = 0; i < generation.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < _SDRNodes.size(); j++) {
            sum += _SDRNodes[j]._weights[i] * _SDRNodes[j]._state;
        }

        generation[i] = sum;
    }
}