using System;
using System.Collections.Generic;

using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNetwork;
using NeuralNetwork.ActivationFunctions;

namespace NeuralNetworkTests
{
    [TestClass]
    public class NodeTests
    {
        [TestMethod]
        public void TanhFunctionReturnsCorrectResult()
        {
            Node node = new Node();
            node.Input = 10;

            node.ActivationFunction = new HyperbolicTangentActivationFunction();

            node.Activate(null);

            Assert.AreEqual(Math.Tanh(node.Input), node.ActivatedSum);
        }

        [TestMethod]
        public void SigmoidFunctionReturnsCorrectResult()
        {
            Node node = new Node();
            node.Input = 10;

            node.ActivationFunction = new LogisticSigmoidActivationFunction();

            node.Activate(null);

            Assert.AreEqual((1.0 / (1.0 + Math.Exp(-(node.Input)))), node.ActivatedSum);
        }

        [TestMethod]
        public void SoftmaxFunctionReturnsCorrectResult()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new SoftmaxActivationFunction()),
                new Node(new SoftmaxActivationFunction()),
                new Node(new SoftmaxActivationFunction())
            };

            nodes[0].Input = 2.0;
            nodes[1].Input = -1.0;
            nodes[2].Input = 4.0;

            Layer layer = new Layer(LayerType.Hidden, new SoftmaxActivationFunction(), 1, 3);
            layer.Nodes = nodes;

            double scalingFactor = Math.Exp(2 - 4) + Math.Exp(-1 - 4) + Math.Exp(4 - 4);
            double result = Math.Exp(2 - 4) / scalingFactor;

            List<double> inputs = new List<double>();
            foreach(Node node in nodes)
            {
                inputs.Add(node.Input);
            }

            layer.ActivateNodes();

            //nodes[0].Softmax(inputs.ToArray());

            Console.WriteLine("expected {0}, actual {1}", result, nodes[0].ActivatedSum);

            Assert.AreEqual(result, nodes[0].ActivatedSum);
        }
    }
}
