using System;
using System.Collections.Generic;
using NeuralNetwork.ActivationFunctions;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<double> Weights { get; set; }
        public List<double> Biases { get; set; }
        public List<Node> Nodes { get; set; }

        public LayerType LayerType { get; set; }

        public IActivationFunction ActivationFunction { get; set; }

        public int LayerOrder {get; set;}

        public Layer(LayerType layerType, IActivationFunction activationFunction, int layerOrder, int numberOfNodes)
        {
            if (layerType == LayerType.Input)
            {
                if (layerOrder != 0)
                {
                    throw new ArgumentException("LayerType of Input must be at LayerOrder of 0.");
                }

                if (activationFunction != null)
                {
                    throw new ArgumentException("LayerType of Input must have ActivationFunction of null.");
                }
            }
            else
            {
                if (layerOrder == 0)
                {
                    throw new ArgumentException("LayerType of Hidden or Output must not be at LayerOrder of 0.");
                }

            }

            if (numberOfNodes <= 0)
            {
                throw new ArgumentException("Must have one or more nodes.");
            }
            
            //assign all variables
            this.Weights = new List<double>();
            this.Biases = new List<double>();
            this.LayerOrder = layerOrder;
            this.LayerType = layerType;
            this.ActivationFunction = activationFunction;

            //generate nodes
            this.Nodes = new List<Node>();
            for (int i = 0; i < numberOfNodes; i++)
            {
                this.Nodes.Add(new Node(activationFunction));
            }
        }

        public void ActivateNodes()
        {
            foreach(Node node in this.Nodes)
            {
                node.Activate(this);
            }

            //in case of softmax
            //List<double> inputs = new List<double>();
            //foreach (Node nd in this.Nodes)
            //{
            //    inputs.Add(nd.Input);
            //}

            //foreach(Node node in this.Nodes)
            //{
            //    switch(ActivationFunction)
            //    {
            //        case ActivationFunction.Sigmoid:
            //            node.Sigmoid();
            //            break;
            //        case ActivationFunction.Tanh:
            //            node.TanH();
            //            break;
            //        case ActivationFunction.Softmax:
            //            node.Softmax(inputs.ToArray());
            //            break;
            //        case ActivationFunction.None:
            //            node.ActivatedSum = node.Input;
            //            break;
            //    }
            //}
        }

        public Layer(List<double> weights, List<double> biases, LayerType layerType, IActivationFunction activationFunction, int layerOrder, List<Node> nodes)
        {
            throw new NotImplementedException();
        }

        public void FeedForward(Layer nextLayer)
        {
            if (this.Biases.Count != nextLayer.Nodes.Count || this.Weights.Count != nextLayer.Nodes.Count * this.Nodes.Count)
            {
                throw new Exception("Wrong number of weights or biases.");
            }

            foreach(Node node in this.Nodes)
            {
                if (node.ActivatedSum == null)
                {
                    throw new Exception("Nodes must be activated before FeedForward is called.");
                }
            }
            
            for(int i = 0; i < nextLayer.Nodes.Count; i++)
            {
                double input = 0;

                for(int p = 0; p < this.Nodes.Count; p++)
                {
                    input += Convert.ToDouble(this.Nodes[p].ActivatedSum) * this.Weights[(i * this.Nodes.Count) + p];
                }

                input += this.Biases[i];

                nextLayer.Nodes[i].Input = input;
            }
        }
    }
}
