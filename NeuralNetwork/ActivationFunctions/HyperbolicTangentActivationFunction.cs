using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ActivationFunctions
{
    public class HyperbolicTangentActivationFunction : IActivationFunction
    {
        public double Activate(Node currentNode, Layer layer)
        {
            if (currentNode == null)
            {
                throw new ArgumentException("Current node cannot be null.");
            }

            return Math.Tanh(currentNode.Input);
        }
    }
}
