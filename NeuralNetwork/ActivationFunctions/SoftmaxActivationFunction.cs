using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ActivationFunctions
{
    public class SoftmaxActivationFunction : IActivationFunction
    {
        public double Activate(Node currentNode, Layer layer)
        {
            if (layer == null)
            {
                throw new ArgumentException("Layer cannot be null.");
            }

            if (currentNode == null)
            {
                throw new ArgumentException("Current node cannot be null.");
            }

            List<double> inputs = layer.Nodes.Select(x => x.Input).ToList();

            double max = inputs.Max();
            double scalingFactor = 0;

            foreach (double input in inputs)
            {
                scalingFactor += Math.Exp(input - max);
            }

            return Math.Exp(currentNode.Input - max) / scalingFactor;
        }
    }
}
