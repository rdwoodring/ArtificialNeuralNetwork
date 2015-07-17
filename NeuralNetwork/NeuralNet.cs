using System;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        public List<Layer> Layers { get; set; }
        public List<double> Weights { get; set; }
        public List<double> Biases { get; set; }

        public NeuralNet(List<Layer> layers, List<double> weights, List<double> biases)
        {
            this.Layers = new List<Layer>(layers);
            this.Weights = new List<double>(weights);
            this.Biases = new List<double>(biases);

            //sort layers ascending by LayerOrder
            this.Layers = this.Layers.OrderBy(i => i.LayerOrder).ToList();

            //portion out biases and weights
            for(int i = 0; i < this.Layers.Count - 1; i++)
            {
                int takeNumBiases = this.Layers[i+1].Nodes.Count;
                int takeNumWeights = this.Layers[i].Nodes.Count * this.Layers[i + 1].Nodes.Count;
                
                this.Layers[i].Biases.AddRange(this.Biases.Take(takeNumBiases));
                this.Biases.RemoveRange(0, takeNumBiases);

                this.Layers[i].Weights.AddRange(this.Weights.Take(takeNumWeights));
                this.Weights.RemoveRange(0, takeNumWeights);
            }
        }

        public void Run()
        {           
            for (int i = 0; i < this.Layers.Count; i++)
            {
                if (this.Layers[i].LayerType == LayerType.Output)
                {
                    this.Layers[i].ActivateNodes();
                }
                else
                {
                    this.Layers[i].ActivateNodes();
                    this.Layers[i].FeedForward(this.Layers[i + 1]);
                }
            }
        }
    }
}
