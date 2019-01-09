#include "fuzzy.h"

#include <plot.h>

void FuzzyNumber::print(const String & name) const
{
    python::Figure p;
    p.addSeries(n);
    p.save(name);
}
