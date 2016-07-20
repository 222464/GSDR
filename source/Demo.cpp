#include "gsdr/GSDR.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

struct Image {
    std::vector<sf::Uint8> _intensities;
};

void loadMNISTimage(std::ifstream &fromFile, int index, Image &img) {
    const int headerSize = 16;
    const int imageSize = 28 * 28;

    fromFile.seekg(headerSize + index * imageSize);

    if (img._intensities.size() != 28 * 28)
        img._intensities.resize(28 * 28);

    fromFile.read(reinterpret_cast<char*>(img._intensities.data()), 28 * 28);
}

int loadMNISTlabel(std::ifstream &fromFile, int index) {
    const int headerSize = 8;

    fromFile.seekg(headerSize + index * 1);

    char label;

    fromFile.read(&label, 1);

    return static_cast<int>(label);
}

int main() {
    std::mt19937 generator(time(nullptr));

    sf::RenderTexture rescaleRT;
    rescaleRT.create(28, 28);

    GSDR gsdr;

    gsdr.createRandom(rescaleRT.getSize().x * rescaleRT.getSize().y, 256, 10, -0.01f, 0.01f, 20.0f, generator);

    // Train on MNIST digits
    std::uniform_int_distribution<int> digitDist(0, 59999);

    std::ifstream fromImageFile("resources/train-images.idx3-ubyte", std::ios::binary | std::ios::in);

    if (!fromImageFile.is_open()) {
        std::cerr << "Could not open train-images.idx3-ubyte!" << std::endl;

        return 1;
    }

    std::ifstream fromLabelFile("resources/train-labels.idx1-ubyte", std::ios::binary | std::ios::in);

    if (!fromLabelFile.is_open()) {
        std::cerr << "Could not open train-labels.idx1-ubyte!" << std::endl;

        return 1;
    }

    sf::Shader sharpenShader;
    sharpenShader.loadFromFile("resources/sharpen.frag", sf::Shader::Fragment);

    const sf::Vector2f upscale(rescaleRT.getSize().x / 28.0f, rescaleRT.getSize().y / 28.0f);

    for (int iter = 0; iter < 8000; iter++) {
        int index = digitDist(generator);

        Image digit;

        loadMNISTimage(fromImageFile, index, digit);

        int label = loadMNISTlabel(fromLabelFile, index);

        // Create image for upscaling
        sf::Image img;
        img.create(28, 28);

        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++) {
                sf::Uint8 c = digit._intensities[x + y * 28];

                img.setPixel(x, y, sf::Color(c, c, c));
            }

        sf::Texture tex;
        tex.loadFromImage(img);
        tex.setSmooth(true);

        rescaleRT.clear();

        sf::Sprite upSprite;
        upSprite.setTexture(tex);
        upSprite.setScale(upscale);

        rescaleRT.draw(upSprite);

        rescaleRT.display();

        sf::Image rImg = rescaleRT.getTexture().copyToImage();

        std::vector<float> input(rescaleRT.getSize().x * rescaleRT.getSize().y);

        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Uint8 c = rImg.getPixel(x, y).r;

                input[x + y * rescaleRT.getSize().x] = c / 255.0f;
            }

        std::vector<float> latent(10, 0.0f);

        latent[label] = 1.0f;

        gsdr.learn(input, latent, 0.0015f, 0.03f);

        if (iter % 10 == 0)
            std::cout << "Iteration " << iter << std::endl;
    }

    // Generate for each label
    /*for (int i = 0; i < 10; i++) {
        std::vector<float> latent(10, 0.0f);

        latent[i] = 1.0f;

        std::vector<float> gen(28 * 28);

        gsdr.generate(latent, gen);

        sf::Image img;

        img.create(28, 28);

        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++) {
                sf::Color c;

                c.r = c.g = c.b = 255 * gen[x + y * 28];

                img.setPixel(x, y, c);
            }

        img.saveToFile("gsdr-gen" + std::to_string(i) + ".png");
    }*/

    sf::RenderWindow window;

    window.create(sf::VideoMode(512, 512), "GSDR Test");

    bool quit = false;

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    sf::Clock clock;

    float dt = 0.017f;

    std::vector<float> latent(10, 0.0f);

    int target = 0;

    latent[target] = 1.0f;

    do {
        clock.restart();

        // ----------------------------- Input -----------------------------

        sf::Event windowEvent;

        while (window.pollEvent(windowEvent))
        {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num0))
            target = 0;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1))
            target = 1;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2))
            target = 2;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
            target = 3;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4))
            target = 4;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5))
            target = 5;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num6))
            target = 6;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num7))
            target = 7;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num8))
            target = 8;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num9))
            target = 9;

        for (int i = 0; i < 10; i++)
            latent[i] += 0.04f * ((i == target ? 1.0f : 0.0f) - latent[i]);

        std::vector<float> gen(rescaleRT.getSize().x * rescaleRT.getSize().y);

        gsdr.generate(latent, gen);

        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = c.g = c.b = 255 * std::min(1.0f, std::max(0.0f, gen[x + y * rescaleRT.getSize().x]));

                img.setPixel(x, y, c);
            }

        window.clear();

        sf::Texture tex;

        tex.loadFromImage(img);
        tex.setSmooth(true);

        sf::Sprite s;

        s.setTexture(tex);

        float scale = window.getSize().x / static_cast<float>(rescaleRT.getSize().x);

        s.setScale(scale, scale);

        sf::RenderStates rs;
        rs.shader = &sharpenShader;

        window.draw(s, rs);

        window.display();
    } while (!quit);

    return 0;
}