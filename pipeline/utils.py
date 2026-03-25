
import torch
def rotation_matrix(angle_deg, axis):
  """
  Creates a rotation matrix for a given angle in degrees and axis.

  Args:
    angle_deg: Rotation angle in degrees.
    axis: Axis of rotation ('x', 'y', or 'z').

  Returns:
    A 3x3 PyTorch tensor representing the rotation matrix.
  """

  theta = torch.deg2rad(torch.tensor(angle_deg))
  c, s = torch.cos(theta), torch.sin(theta)
  if axis == 'x':
    return torch.tensor([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
  elif axis == 'y':
    return torch.tensor([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
  elif axis == 'z':
    return torch.tensor([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])