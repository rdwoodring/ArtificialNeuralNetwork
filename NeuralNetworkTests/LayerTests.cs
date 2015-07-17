using System;
using System.Collections.Generic;

using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNetwork;
using NeuralNetwork.ActivationFunctions;

namespace NeuralNetworkTests
{
    [TestClass]
    public class LayerTests
    {
        [TestMethod]
        public void ActivateSigmoidSettingNodeValuesCorrectly()
        {
            Layer layer = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 3);

            layer.Nodes[0].Input = 2.0;
            layer.Nodes[1].Input = -1.0;
            layer.Nodes[2].Input = 4.0;

            double result1 = (1.0 / (1.0 + Math.Exp(-(layer.Nodes[0].Input))));
            double result2 = (1.0 / (1.0 + Math.Exp(-(layer.Nodes[1].Input))));
            double result3 = (1.0 / (1.0 + Math.Exp(-(layer.Nodes[2].Input))));

            layer.ActivateNodes();

            Assert.AreEqual(result1, layer.Nodes[0].ActivatedSum);
            Assert.AreEqual(result2, layer.Nodes[1].ActivatedSum);
            Assert.AreEqual(result3, layer.Nodes[2].ActivatedSum);
        }

        [TestMethod]
        public void ActivateTanhSettingNodeValuesCorrectly()
        {
            Layer layer = new Layer(LayerType.Hidden, new HyperbolicTangentActivationFunction(), 1, 3);

            layer.Nodes[0].Input = 2.0;
            layer.Nodes[1].Input = -1.0;
            layer.Nodes[2].Input = 4.0;

            double result1 = Math.Tanh(layer.Nodes[0].Input);
            double result2 = Math.Tanh(layer.Nodes[1].Input);
            double result3 = Math.Tanh(layer.Nodes[2].Input);

            layer.ActivateNodes();

            Assert.AreEqual(result1, layer.Nodes[0].ActivatedSum);
            Assert.AreEqual(result2, layer.Nodes[1].ActivatedSum);
            Assert.AreEqual(result3, layer.Nodes[2].ActivatedSum);
        }

        [TestMethod]
        public void ActivateSoftmaxSettingNodeValuesCorrectly()
        {
            Layer layer = new Layer(LayerType.Hidden, new SoftmaxActivationFunction(), 1, 3);

            layer.Nodes[0].Input = 2.0;
            layer.Nodes[1].Input = -1.0;
            layer.Nodes[2].Input = 4.0;

            double scalingFactor = Math.Exp(2 - 4) + Math.Exp(-1 - 4) + Math.Exp(4 - 4);
            double result1 = Math.Exp(2 - 4) / scalingFactor;
            double result2 = Math.Exp(-1 - 4) / scalingFactor;
            double result3 = Math.Exp(4 - 4) / scalingFactor;

            layer.ActivateNodes();

            Assert.AreEqual(result1, layer.Nodes[0].ActivatedSum);
            Assert.AreEqual(result2, layer.Nodes[1].ActivatedSum);
            Assert.AreEqual(result3, layer.Nodes[2].ActivatedSum);
        }

        [TestMethod]
        public void LayerConstructorAllowsLayersWithActivationTypeNone()
        {
            Layer input = new Layer(LayerType.Input, null, 0, 4);
            Layer hidden = new Layer(LayerType.Hidden, null, 1, 4);
            Layer output = new Layer(LayerType.Output, null, 2, 4);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void FeedForwardThrowsIfLayerActivatedSumsAreNull()
        {            
            Layer hidden = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);
            Layer output = new Layer(LayerType.Output, new SoftmaxActivationFunction(), 2, 3);

            hidden.Nodes[0].Input = 1;
            hidden.Nodes[1].Input = 2;
            hidden.Nodes[2].Input = 3;
            hidden.Nodes[3].Input = 4;

            hidden.FeedForward(hidden);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void FeedForwardThrowsIfBiasesDoesNotEqualNextLayerNodes()
        {
            Layer input = new Layer(LayerType.Input, null, 0, 5);
            Layer hidden = new Layer( LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);

            input.Nodes[0].Input = 1;
            input.Nodes[1].Input = 2;
            input.Nodes[2].Input = 3;
            input.Nodes[3].Input = 4;

            input.Nodes[0].ActivatedSum = 1;
            input.Nodes[1].ActivatedSum = 2;
            input.Nodes[2].ActivatedSum = 3;
            input.Nodes[3].ActivatedSum = 4;

            input.ActivateNodes();

            input.FeedForward(hidden);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LayerConstructorThrowsIfLayerTypeNotInputAndLayerOrderZero()
        {
            Layer layer = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 0, 5);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LayerConstructorThrowsIfLayerTypeInputAndLayerOrderNotZero()
        {
            Layer layer = new Layer(LayerType.Input, null, 1, 5);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LayerConstructorThrowsIfLayerTypeInputAndActivationFunctionNotNull()
        {
            Layer layer = new Layer(LayerType.Input, new LogisticSigmoidActivationFunction(), 0, 5);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LayerConstructorThrowsIfNumberOfNodesLessThanOne()
        {
            Layer layer = new Layer(LayerType.Input, null, 0, 0);
        }

        [TestMethod]
        public void LayerConstructorCreatesCorrectNumberOfNodes()
        {
            Layer layer = new Layer(LayerType.Input, null, 0, 4);

            Assert.AreEqual(4, layer.Nodes.Count);
        }

        [TestMethod]
        public void FeedForwardSetsNextLayerInputsCorrectly()
        {
            List<double> biases = new List<double>()
            {
                1,
                1,
                1,
                1
            };

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
                1
            };
            
            Layer input = new Layer(LayerType.Input, null, 0, 3);
            Layer hidden = new Layer(LayerType.Hidden, new LogisticSigmoidActivationFunction(), 1, 4);

            input.Weights = weights;
            input.Biases = biases;

            input.Nodes[0].Input = 1;
            input.Nodes[1].Input = 2;
            input.Nodes[2].Input = 3;

            input.Nodes[0].ActivatedSum = 1;
            input.Nodes[1].ActivatedSum = 2;
            input.Nodes[2].ActivatedSum = 3;

            input.ActivateNodes();

            input.FeedForward(hidden);

            Assert.AreEqual(7, hidden.Nodes[0].Input);
        }
    }
}
