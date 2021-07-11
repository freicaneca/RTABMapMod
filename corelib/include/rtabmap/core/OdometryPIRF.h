#ifndef ODOMETRYPIRF_H_
#define ODOMETRYPIRF_H_

#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/Signature.h>

namespace rtabmap {
    
class Registration;

class RTABMAP_EXP OdometryPIRF : public Odometry
{
public:
	OdometryPIRF(const rtabmap::ParametersMap & parameters = rtabmap::ParametersMap());
	virtual ~OdometryPIRF();

	virtual void reset(const Transform & initialPose = Transform::getIdentity());

	const Signature & getRefFrame() const {return refFrame_;}

	virtual Odometry::Type getType() {return Odometry::kTypePIRF;}

//private:

private:

        int w;

        std::vector<Signature> frames;

};
#endif
