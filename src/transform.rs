use crate::{matrix::Mat4, quad::Quat, vec::Vec3};

#[derive(Debug)]
pub struct Transform {
    pub position: Vec3<f32>,
    pub rotation: Quat<f32>,
    pub scale: Vec3<f32>,
}

impl Transform {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            position: Vec3::splat(0.0),
            rotation: Quat::from_axis_angle(Vec3::forward(), 0.0),
            scale: Vec3::splat(1.0),
        }
    }

    pub const fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position.x = x;
        self.position.y = y;
        self.position.z = z;
    }

    pub const fn set_scale(&mut self, x: f32, y: f32, z: f32) {
        self.scale.x = x;
        self.scale.y = y;
        self.scale.z = z;
    }

    #[inline]
    #[must_use]
    pub fn as_mat4(&self) -> Mat4<f32> {
        let mat = self.rotation.to_rotation_matrix4();
        let pos = self.position_matrix();

        mat.multiply(&pos)
    }

    #[inline]
    const fn position_matrix(&self) -> Mat4<f32> {
        Mat4::from_array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [self.position.x, self.position.y, self.position.z, 1.0],
        ])
    }
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn test() {
    let mut transform = Transform::new();
    transform.position += Vec3::splat(2.0);
    transform.scale = Vec3::splat(3.0);
    let mat = transform.as_mat4();

    println!("{mat:?}");
}
