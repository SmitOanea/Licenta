using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace colturi
{
    class Cub
    {
        public Colt[] clt;
        public Muchie[] mch;
        public Cub()
        {
            clt = new Colt[10];
            for (int i = 0; i <= 9; i++)
                clt[i] = new Colt();
            mch = new Muchie[15];
            for (int i = 0; i <= 14; ++i)
                mch[i] = new Muchie();
        }
    }
}


