package org.example.clipBoard;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.image.BufferedImage;

public class CopyToClipBoard {

    public static void copyToClipBoard(BufferedImage image){
        try {
            //Віддаємо зображення в наш клас для перевірок посля чого щоб буфер обміну міг працювати
            ImageSelectWrapper SelectionWrapper = new ImageSelectWrapper(image);
            Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard(); //отримуємо системний буфер обміну
            clipboard.setContents(SelectionWrapper, null); // перекладуємо у буфер обміну зображення

        }catch (Exception e){
            System.out.println(e.getMessage());
        }

    }




}
