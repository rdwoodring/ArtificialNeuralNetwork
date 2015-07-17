using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.ActivationFunctions;
using NeuralNetwork;

namespace NeuralNetworkTests
{
    [TestClass]
    public class NeuralNetTests
    {
        [TestMethod]
        public void NeuralNetConstructorOrdersLayersAscendingByLayerOrder()
        {
            Layer input = new Layer(LayerType.Input, null, 0, 3);
            Layer output = new Layer(LayerType.Output, new SoftmaxActivationFunction(), 2, 2);
            Layer hidden = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);            

            input.Nodes[0].Input = 1;
            input.Nodes[1].Input = 2;
            input.Nodes[2].Input = 3;

            input.Nodes[0].ActivatedSum = 1;
            input.Nodes[1].ActivatedSum = 2;
            input.Nodes[2].ActivatedSum = 3;

            List<Layer> layers = new List<Layer>();
            layers.Add(input);
            layers.Add(output);
            layers.Add(hidden);

            //24 weights
            List<double> weights = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            };

            //6 biases
            List<double> biases = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1
            };

            NeuralNet nn = new NeuralNet(layers, weights, biases);

            //nn.Run();

            Assert.AreEqual(0, nn.Layers[0].LayerOrder);
            Assert.AreEqual(1, nn.Layers[1].LayerOrder);
            Assert.AreEqual(2, nn.Layers[2].LayerOrder);
        }

        [TestMethod]
        public void NeuralNetConstructorAssignsCorrectNumberOfBiasesToEachLayer()
        {
            Layer input = new Layer(LayerType.Input, null, 0, 3);
            Layer hidden = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);
            Layer output = new Layer( LayerType.Output, new SoftmaxActivationFunction(), 2, 2);

            input.Nodes[0].Input = 1;
            input.Nodes[1].Input = 2;
            input.Nodes[2].Input = 3;

            input.Nodes[0].ActivatedSum = 1;
            input.Nodes[1].ActivatedSum = 2;
            input.Nodes[2].ActivatedSum = 3;

            List<Layer> layers = new List<Layer>();
            layers.Add(input);
            layers.Add(hidden);
            layers.Add(output);

            //24 weights
            List<double> weights = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            };

            //6 biases
            List<double> biases = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1
            };

            NeuralNet nn = new NeuralNet(layers, weights, biases);

            Assert.AreEqual(nn.Layers[1].Nodes.Count, nn.Layers[0].Biases.Count);
            Assert.AreEqual(nn.Layers[2].Nodes.Count, nn.Layers[1].Biases.Count);
        }

        [TestMethod]
        public void NeuralNetConstructorAssignsCorrectNumberOfWeightsToEachLayer()
        {
            Layer input = new Layer(LayerType.Input, null, 0, 3);
            Layer hidden = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);
            Layer output = new Layer(LayerType.Output, new SoftmaxActivationFunction(), 2, 2);

            input.Nodes[0].Input = 1;
            input.Nodes[1].Input = 2;
            input.Nodes[2].Input = 3;

            input.Nodes[0].ActivatedSum = 1;
            input.Nodes[1].ActivatedSum = 2;
            input.Nodes[2].ActivatedSum = 3;

            List<Layer> layers = new List<Layer>();
            layers.Add(input);
            layers.Add(hidden);
            layers.Add(output);

            //24 weights
            List<double> weights = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            };

            //6 biases
            List<double> biases = new List<double>()
            {
                1,
                1,
                1,
                1,
                1,
                1
            };

            NeuralNet nn = new NeuralNet(layers, weights, biases);

            Assert.AreEqual(nn.Layers[0].Nodes.Count * nn.Layers[1].Nodes.Count, nn.Layers[0].Weights.Count);
            Assert.AreEqual(nn.Layers[1].Nodes.Count * nn.Layers[2].Nodes.Count, nn.Layers[1].Weights.Count);
        }
    }
}
