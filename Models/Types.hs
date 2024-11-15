module Models.Types where
import Data.Matrix as M

-- type alias for data
type Data = (Matrix Double, Matrix Double)

-- Type alias for models
type Model = Matrix Double -> Data -> Int -> Weights

-- Type alias for weights
type Weights = IO (Matrix Double)
