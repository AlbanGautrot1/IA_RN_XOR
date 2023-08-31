#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>


double sigmoid(double x) {
    return 1.0f / (1.0f + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0f - x);
}

double init_weight() {
    return (double) rand() / ((double)RAND_MAX);
}

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


#define numInput 2
#define numHidden 2
#define numOutput 1
#define trainingSetsNum 4

int main() {

    const double learningRate = 0.1f;
    double hiddenLayer[numHidden];
    double outputLayer[numOutput];

    double hiddenLayerBias[numHidden];
    double outputLayerBias[numOutput];

    double hiddenWeights[numInput][numHidden];
    double outputWeights[numHidden][numOutput];

    double traningInput[trainingSetsNum][numInput] = {{0.0f, 0.0f},
                                                      {0.0f, 1.0f},
                                                      {1.0f, 0.0f},
                                                      {1.0f, 1.0f}};

    double traningOutput[trainingSetsNum][numOutput] = {{0.0f},
                                                        {1.0f},
                                                        {1.0f},
                                                        {0.0f}};
    for (int i = 0; i < numInput; i++) {
        for (int j = 0; j < numHidden; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }

    for (int i = 0; i < numHidden; i++) {
        for (int j = 0; j < numOutput; j++) {
            outputWeights[i][j] = init_weight();
        }
    }

    for (int i = 0; i < numOutput ; ++i) {
       outputLayerBias[i] = init_weight();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};
    int epochNumb = 100000;

    // Training
    for (int epoch = 0; epoch < epochNumb; epoch++) {
        shuffle(trainingSetOrder, trainingSetsNum);

        for (int x = 0; x < trainingSetsNum; x++) {
            int i = trainingSetOrder[x];

            // Forward
            // Hidden layer activation
            for (int j = 0; j < numHidden; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInput; k++) {
                    activation += traningInput[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            // output activation
            for (int j = 0; j < numOutput; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numHidden; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            //printf("Input: %g %g Output: %g Expected: %g\n", traningInput[i][0], traningInput[i][1], outputLayer[0], traningOutput[i][0]);

            // Backward
            // compute change in output weights
            double deltaOutput[numOutput];
            for (int j = 0; j < numOutput; j++) {
                double error = (traningOutput[i][j] - outputLayer[j]);
                deltaOutput[j] = error * sigmoidDerivative(outputLayer[j]);
            }

            // compute change in hidden weights
            double deltaHidden[numHidden];
            for (int j = 0; j < numHidden; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutput; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * sigmoidDerivative(hiddenLayer[j]);
            }

            // update output weights
            for (int j = 0; j < numOutput; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k = 0; k < numHidden; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
                }
            }

            // update hidden weights
            for (int j = 0; j < numHidden; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for (int k = 0; k < numInput; k++) {
                    hiddenWeights[k][j] += traningInput[i][k] * deltaHidden[j] * learningRate;
                }
            }

        }
    }
    // Print final weights after training
    fputs ("Final Hidden Weights\n[ ", stdout);
    for (int j=0; j<numHidden; j++) {
        fputs ("[ ", stdout);
        for(int k=0; k<numInput; k++) {
            printf ("%f ", hiddenWeights[k][j]);
        }
        fputs ("] ", stdout);
    }

    fputs ("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j=0; j<numHidden; j++) {
        printf ("%f ", hiddenLayerBias[j]);
    }

    fputs ("]\nFinal Output Weights", stdout);
    for (int j=0; j<numOutput; j++) {
        fputs ("[ ", stdout);
        for (int k=0; k<numHidden; k++) {
            printf ("%f ", outputWeights[k][j]);
        }
        fputs ("]\n", stdout);
    }

    fputs ("Final Output Biases\n[ ", stdout);
    for (int j=0; j<numOutput; j++) {
        printf ("%f ", outputLayerBias[j]);

    }

    fputs ("]\n", stdout);
    // store final weights in a variable



    // store biases in a variable

    // Test
    double testinput [trainingSetsNum][numInput] = {{0.0f, 0.0f},
                                                    {0.0f, 1.f},
                                                    {1.0f, 0.0f},
                                                    {1.0f, 1.0f}};
    for (int i = 0; i < trainingSetsNum; i++) {
        // Hidden layer activation
        for (int j = 0; j < numHidden; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInput; k++) {
                activation += testinput[i][k] * hiddenWeights[k][j];
            }
            hiddenLayer[j] = sigmoid(activation);
        }
        // output activation
        for (int j = 0; j < numOutput; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numHidden; k++) {
                activation += hiddenLayer[k] * outputWeights[k][j];
            }
            outputLayer[j] = sigmoid(activation);
        }

        printf("Input: %g %g Output: %g Expected: %g\n", testinput[i][0], testinput[i][1], outputLayer[0], traningOutput[i][0]);
    }
    //window
    //create sfml window
#define Height 800
#define Width 800
    sf::RenderWindow window(sf::VideoMode(Height, Width), "SFML works!");
    sf::Event event;
    if (!sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
        while (!sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            //wait
        }
    }


    float resolution = 5;
    float cols = Height / resolution;
    float rows = Width / resolution;


    //keep window open until closed
    while (window.isOpen())
    {
        //check for events
        while (window.pollEvent(event))
        {
            //close window if close button is pressed
            if (event.type == sf::Event::Closed)
                window.close();
        }
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                double x1 = (i / (double)cols);
                double x2 = (j / (double)rows);
                //input
                double inputdraw [1][numInput] = {x1, x2};
                    // Hidden layer activation
                    for (int j = 0; j < numHidden; j++) {
                        double activation = hiddenLayerBias[j];
                        for (int k = 0; k < numInput; k++) {
                            activation += inputdraw[0][k] * hiddenWeights[k][j];
                        }
                        hiddenLayer[j] = sigmoid(activation);
                    }
                    // output activation
                    for (int m = 0; m < numOutput; m++) {
                        double activation = hiddenLayerBias[m];
                        for (int k = 0; k < numHidden; k++) {
                            activation += hiddenLayer[k] * outputWeights[k][m];
                        }
                        outputLayer[m] = sigmoid(activation);
                    }
                    double y = outputLayer[0];
                    sf::RectangleShape rect;
                    rect.setSize(sf::Vector2f(resolution,2* resolution));
                    rect.setPosition(i * resolution, j * resolution);
                    rect.setFillColor(sf::Color(y * 255, y * 255, y * 255));
                    window.draw(rect);
                    window.display();
                }
            //pause program
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
                while (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
                    //wait
                }
            }
        }
    }
    return 0;
}
