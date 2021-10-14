#ifndef POINT_OS1_H_
#define POINT_OS1_H_

#include <pcl/point_types.h>

struct EIGEN_ALIGN16 PointOS1 {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static inline PointOS1 make(float x, float y, float z, float intensity,
                                uint32_t t, uint16_t reflectivity, uint8_t ring,
                                uint16_t noise, uint32_t range) {
        return {x, y, z, 0.0, intensity, t, reflectivity, ring, noise, range};
    }
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointOS1,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, t, t)
    (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring)
    (uint16_t, noise, noise)
    (uint32_t, range, range)
)

#endif // POINT_OS1_H_
